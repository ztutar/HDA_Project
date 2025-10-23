#TODO: add docstring explanation for this module. write in details and explain each function


from typing import Tuple
from dataclasses import asdict
import logging
import os
import csv
import json
import gc
import io
import numpy as np
import tensorflow as tf
import keras

from BoneAgePrediction.utils.logger import get_logger, setup_logging 
from BoneAgePrediction.utils.seeds import set_seeds
from BoneAgePrediction.utils.config import load_config
from BoneAgePrediction.utils.path_manager import incremental_path

from BoneAgePrediction.data.dataset_loader import make_dataset

from BoneAgePrediction.models.B0_global_cnn import build_GlobalCNN

from BoneAgePrediction.training.losses import get_loss
from BoneAgePrediction.training.metrics import mae, rmse, count_params, estimate_gmacs, EpochTimer
from BoneAgePrediction.training.callbacks import make_callbacks

logger = get_logger(__name__)


def _ensure_logging() -> None:
   """Initialize logging to experiments/logs if no handlers configured yet."""
   if not logging.getLogger().handlers:
      setup_logging(log_dir=os.path.join("experiments", "logs"))


def train_GlobalCNN(config_path: str) -> Tuple[keras.Model, keras.callbacks.History]:
   #TODO: add dpcstring explanation for this function. write in details and explain each step
   
   # -----------------------
   # Config & reproducibility
   # -----------------------
   _ensure_logging()
   config_bundle = load_config(config_path)
   data_cfg = config_bundle.data
   model_cfg = config_bundle.model
   training_cfg = config_bundle.training
   optimizer_cfg = training_cfg.optimizer
   set_seeds()
   
   config_filename = os.path.basename(config_path) if config_path else "default"
   config_dict = asdict(config_bundle)
   config_params = json.dumps(config_bundle.raw, sort_keys=True)
   model_name = "B0_GlobalCNN"
   logger.info(f"Starting {model_name} training using config %s", config_filename)
   logger.debug("Configuration parameters: %s", config_dict)
   
   # -----------------------
   # Data
   # -----------------------
   data_path = data_cfg.data_path
   target_h = data_cfg.target_h
   target_w = data_cfg.target_w
   keep_aspect_ratio = data_cfg.keep_aspect_ratio
   pad_value = data_cfg.pad_value
   batch_size = data_cfg.batch_size
   clahe = data_cfg.clahe
   augment = data_cfg.augment
   cache = data_cfg.cache
   
   train_ds = make_dataset(
      data_path=data_path,
      split='train',
      target_h=target_h,
      target_w=target_w,
      keep_aspect_ratio=keep_aspect_ratio,
      pad_value=pad_value,
      batch_size=batch_size,
      clahe=clahe,
      augment=augment,
      cache=cache,
   )
   
   val_ds = make_dataset(
      data_path=data_path,
      split='val',
      target_h=target_h,
      target_w=target_w,
      keep_aspect_ratio=keep_aspect_ratio,
      pad_value=pad_value,
      batch_size=batch_size,
      clahe=clahe,
      augment=False,
      cache=cache,
   )
   logger.info("Prepared training and validation datasets from %s", data_path)

   def _select_image(features: dict, label: tf.Tensor):
      return features["image"], label

   train_ds = train_ds.map(_select_image, num_parallel_calls=tf.data.AUTOTUNE)
   train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

   val_ds = val_ds.map(_select_image, num_parallel_calls=tf.data.AUTOTUNE)
   val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
   
   # -----------------------
   # Model
   # -----------------------
   channels = model_cfg.channels
   num_blocks = model_cfg.num_blocks
   dense_units = model_cfg.dense_units
   input_shape = (target_h, target_w, 1)  # grayscale input
   
   model = build_GlobalCNN(
      input_shape=input_shape,
      num_blocks=num_blocks,
      channels=channels,
      dense_units=dense_units,
   )
   logger.info(
      "%s architecture: %d blocks, channels=%s, dense_units=%d",
      model_name,
      num_blocks,
      channels,
      dense_units,
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
   # learning rate schedule
   reduce_lr = keras.callbacks.ReduceLROnPlateau(
      monitor='val_mae', factor=0.1,
      patience=4, min_lr=0.0001, verbose=1)
   callbacks.append(reduce_lr)
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
   gmacs = estimate_gmacs(model, input_shape)
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
   
   model.summary()
   summary_stream = io.StringIO()
   model.summary(print_fn=lambda line: summary_stream.write(line + "\n"))
   logger.info("Model summary:\n%s", summary_stream.getvalue())
   
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
