#TODO: add docstring explanation for this module. write in details and explain each function


from typing import Tuple, Dict
from dataclasses import asdict
import logging
import os
import csv
import json
import gc
import numpy as np
import tensorflow as tf
import keras

from BoneAgePrediction.utils.logger import get_logger, setup_logging 
from BoneAgePrediction.utils.seeds import set_seeds
from BoneAgePrediction.utils.config import load_config
from BoneAgePrediction.utils.path_manager import incremental_path

from BoneAgePrediction.data.dataset_loader import make_roi_dataset

from BoneAgePrediction.roi.roi_locator import train_locator_and_save_rois

from BoneAgePrediction.models.R1_roi_cnn import build_ROI_CNN

from BoneAgePrediction.training.losses import get_loss
from BoneAgePrediction.training.metrics import mae, rmse, count_params, estimate_gmacs, EpochTimer
from BoneAgePrediction.training.callbacks import make_callbacks

logger = get_logger(__name__)


def _ensure_logging() -> None:
   """Initialize logging to experiments/logs if no handlers configured yet."""
   if not logging.getLogger().handlers:
      setup_logging(log_dir=os.path.join("experiments", "logs"))


def train_ROI_CNN(config_path: str) -> Tuple[keras.Model, keras.callbacks.History]:
   #TODO: add docstring explanation for this function. write in details and explain each step
   """
   Orchestrate ROI-only training:
   1) Train a locator and generate ROI crops (if not already present).
   2) Train R1 ROI-only model on saved crops.
   """
   
   
   # -----------------------
   # Config & reproducibility
   # -----------------------
   _ensure_logging()
   config_bundle = load_config(config_path)
   data_cfg = config_bundle.data
   
   roi_cfg = config_bundle.roi
   locator_cfg = roi_cfg.locator
   extractor_cfg = roi_cfg.extractor
   
   model_cfg = config_bundle.model
   training_cfg = config_bundle.training
   optimizer_cfg = training_cfg.optimizer
   set_seeds()
   
   config_filename = os.path.basename(config_path) if config_path else "default"
   config_dict = asdict(config_bundle)
   config_params = json.dumps(config_bundle.raw, sort_keys=True)
   model_name = "R1_ROI_CNN"
   logger.info(f"Starting {model_name} training using config %s", config_filename)
   logger.debug("Configuration parameters: %s", config_dict)


   # -----------------------
   # ROI Locator & Extractor
   # -----------------------
   roi_path = locator_cfg.roi_path
   os.makedirs(roi_path, exist_ok=True)

   # Generate crops for each split (train/val/test)
   for split in ["train", "validation", "test"]:
      split_dir = os.path.join(roi_path, split)
      carpal_dir = os.path.join(split_dir, "carpal")
      metaph_dir = os.path.join(split_dir, "metaph")
      if not (os.path.exists(carpal_dir) and os.path.exists(metaph_dir)):
         logger.info(f"[R1] Generating ROIs for {split} ...")
         train_locator_and_save_rois(
            config = config_bundle,
            split = split,
            out_root = roi_path,
            save_heatmaps = bool(extractor_cfg.save_heatmaps, False),
         )


   # -----------------------
   # ROI Datasets
   # -----------------------
   data_path = data_cfg.data_path
   batch_size = data_cfg.batch_size
   cache = data_cfg.cache
   
   train_ds = make_roi_dataset(
      data_path = data_path,
      roi_path = roi_path,
      split = 'train',
      batch_size = batch_size,
      cache = cache,
   )
   
   val_ds = make_roi_dataset(
      data_path = data_path,
      roi_path = roi_path,
      split = 'validation',
      batch_size = batch_size,
      cache = cache,
   )
   logger.info("Prepared training and validation ROI datasets from %s", data_path)

   def _select_model_inputs(features, label):
      inputs = {
         "carpal": features["carpal"],
         "metaph": features["metaph"],
      }
      if model_cfg.use_gender:
         inputs["gender"] = tf.cast(features["gender"], tf.int32)
      return inputs, label

   train_ds = train_ds.map(_select_model_inputs, num_parallel_calls=tf.data.AUTOTUNE)
   val_ds = val_ds.map(_select_model_inputs, num_parallel_calls=tf.data.AUTOTUNE)
   train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
   val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
   
   # Deduce ROI shape from one batch
   sample_batch = next(iter(train_ds.take(1)))
   roi_shape = tuple(sample_batch[0]["carpal"].shape[1:]) # [H, W, 1]
   
   # -----------------------
   # Model
   # -----------------------
   channels = model_cfg.channels
   dense_units = model_cfg.dense_units
   use_gender = model_cfg.use_gender
   
   model = build_ROI_CNN(
      roi_shape=roi_shape,
      channels=channels,
      dense_units=dense_units,
      use_gender=use_gender,
   )
   
   logger.info(
      "%s architecture: roi_shape=%s, channels=%s, dense_units=%d, use_gender=%s",
      model_name,
      channels,
      dense_units,
      use_gender,
   )
   
   # -----------------------
   # Optimizer (Adam, fixed LR)
   # -----------------------
   learning_rate = optimizer_cfg.learning_rate
   beta_1 = optimizer_cfg.beta_1 # Exponential decay rate for the 1st moment estimates.
   beta_2 = optimizer_cfg.beta_2 # Exponential decay rate for the 2nd moment estimates.
   epsilon = optimizer_cfg.epsilon # Small constant for numerical stability.
   optimizer = keras.optimizers.Adam(
      learning_rate=learning_rate,
      beta_1=beta_1,
      beta_2=beta_2,
      epsilon=epsilon,
   )
   
   # -----------------------
   # Loss & Metrics
   # -----------------------
   loss_name = training_cfg.loss
   delta = training_cfg.huber_delta # only for Huber loss
   loss_fn = get_loss(
      loss_name=loss_name,
      delta=delta,
   )
   
   # -----------------------
   # Compile model
   # -----------------------
   model.compile(
      optimizer=optimizer,
      loss=loss_fn,
      metrics=[mae(), rmse()],
   )
   logger.info("Model compiled with %s loss", loss_name)
   
   # -----------------------
   # Callbacks
   # -----------------------
   save_dir = incremental_path(
      save_dir=training_cfg.save_dir,
      model_name=model_name,
      config_name=os.path.splitext(config_filename)[0],
   )
   patience = training_cfg.patience
   callbacks = make_callbacks(
      save_dir=save_dir,
      model_name=model_name,
      patience=patience,
   )
   epoch_timer = EpochTimer()
   callbacks.append(epoch_timer)
   logger.info("Configured training callbacks with patience: %d", patience)
   
   # -----------------------
   # Train
   # -----------------------
   epochs = training_cfg.epochs
   logger.info("Starting training for %d epochs", epochs)
   history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=callbacks,
      verbose=1,
   )
   logger.info("Training finished") 

   
   # -----------------------
   # Complexity & Timing
   # -----------------------
   num_params = count_params(model)
   gmacs = estimate_gmacs(model, roi_shape)
   epoch_times = epoch_timer.epoch_times
   avg_epoch_time = np.mean(epoch_times) if epoch_times else None
   print(f"[{model_name}] Number of Params: {num_params:,} | GMACs(est): {gmacs:.3f} | Avg epoch time: {avg_epoch_time:.2f}s")
   avg_time_display = f"{avg_epoch_time:.2f}" if avg_epoch_time is not None else "n/a"
   logger.info(
      "[%s] Params: %d | GMACs(est): %.3f | Avg epoch time: %s s",
      model_name,
      num_params,
      gmacs,
      avg_time_display,
   )
      
   # -----------------------
   # Summary CSV
   # -----------------------
   results_csv = training_cfg.results_csv
   os.makedirs(os.path.dirname(results_csv), exist_ok=True)
   header = [
      "model_name",
      "num_params",
      "gmacs",
      "avg_epoch_time_s",
      "best_train_mae",
      "best_train_rmse",
      "best_val_mae",
      "best_val_rmse",
      "config_file",
      "config_params",
   ]
   
   best_train_mae = float(np.min(history.history.get("mae", [np.inf])))
   best_train_rmse = float(np.min(history.history.get("rmse", [np.inf])))
   best_val_mae = float(np.min(history.history.get("val_mae", [np.inf])))
   best_val_rmse = float(np.min(history.history.get("val_rmse", [np.inf])))
   logger.info(
      "Best metrics — train MAE: %.4f, train RMSE: %.4f, val MAE: %.4f, val RMSE: %.4f",
      best_train_mae,
      best_train_rmse,
      best_val_mae,
      best_val_rmse,
   )
   
   write_header = not os.path.exists(results_csv)
   with open(results_csv, "a", newline="") as f:
      w = csv.writer(f)
      if write_header:
         w.writerow(header)
      w.writerow([
         model_name,
         num_params,
         f"{gmacs:.3f}",
         f"{avg_epoch_time:.2f}",
         f"{best_train_mae:.4f}",
         f"{best_train_rmse:.4f}",
         f"{best_val_mae:.4f}",
         f"{best_val_rmse:.4f}",
         config_filename,
         config_params,
      ])
   logger.info("Appended training summary to %s", results_csv)
   
   # -----------------------
   # Cleanup
   # -----------------------
   train_ds = val_ds = None  # drop strong refs before cleanup
   keras.backend.clear_session()
   gc.collect()
   logger.info("Cleaned up after training")
      
   return model, history
