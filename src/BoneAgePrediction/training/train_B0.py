#TODO: add docstring explanation for this module. write in details and explain each function


from typing import Tuple
from dataclasses import asdict
import logging
import os
import gc
import io
import numpy as np
import tensorflow as tf
import keras

from BoneAgePrediction.utils.logger import get_logger, setup_logging, mirror_keras_stdout_to_file
from BoneAgePrediction.utils.seeds import set_seeds
from BoneAgePrediction.utils.config import load_config
from BoneAgePrediction.utils.path_manager import incremental_path

from BoneAgePrediction.data.dataset_loader import make_dataset

from BoneAgePrediction.models.B0_global_cnn import build_GlobalCNN

from BoneAgePrediction.training.losses import get_loss
from BoneAgePrediction.training.metrics import mae, rmse, count_params, EpochTimer
from BoneAgePrediction.training.callbacks import make_callbacks
from BoneAgePrediction.training.summary import append_summary_row

logger = get_logger(__name__)


def _ensure_logging() -> None:
   """Initialize logging to experiments/logs if no handlers configured yet."""
   if not logging.getLogger().handlers:
      setup_logging(log_dir=os.path.join("experiments", "logs"))
   mirror_keras_stdout_to_file()


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
   dense_units = model_cfg.dense_units
   input_shape = (target_h, target_w, 1)  # grayscale input
   
   model = build_GlobalCNN(
      input_shape=input_shape,
      channels=channels,
      dense_units=dense_units,
      name=model_name,
   )
   logger.info(
      "%s architecture: %d blocks, channels=%s, dense_units=%d",
      model_name,
      len(channels),
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
   epoch_times = epoch_timer.epoch_times
   avg_epoch_time = np.mean(epoch_times) if epoch_times else None
   print(f"[{model_name}] Number of Params: {num_params:,} Avg epoch time: {avg_epoch_time:.2f}s")
   avg_time_display = f"{avg_epoch_time:.2f}" if avg_epoch_time is not None else "n/a"
   logger.info(
      "[%s] Params: %d | Avg epoch time: %s s",
      model_name,
      num_params,
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
   val_mae_history = history.history.get("val_mae", [])
   best_epoch_idx = int(np.argmin(val_mae_history)) if val_mae_history else None

   def metric_at(name: str, idx: int, default: float = float("nan")) -> float:
      series = history.history.get(name)
      if series is None or idx is None or idx >= len(series):
         return default
      return float(series[idx])

   train_mae = metric_at("mae", best_epoch_idx)
   train_rmse = metric_at("rmse", best_epoch_idx)
   val_mae = metric_at("val_mae", best_epoch_idx)
   val_rmse = metric_at("val_rmse", best_epoch_idx)

   logger.info(
      "Best epoch metrics — train MAE: %.4f, train RMSE: %.4f, val MAE: %.4f, val RMSE: %.4f",
      train_mae,
      train_rmse,
      val_mae,
      val_rmse,
   )
   
   summary_base = {
      "model_name": model_name,
      "num_params": num_params,
      "avg_epoch_time_s": avg_time_display,
      "train_mae": f"{train_mae:.4f}",
      "train_rmse": f"{train_rmse:.4f}",
      "val_mae": f"{val_mae:.4f}",
      "val_rmse": f"{val_rmse:.4f}",
   }
   append_summary_row(
      results_csv=results_csv,
      base_data=summary_base,
      config_bundle=config_bundle,
      config_filename=config_filename,
   )
   logger.info("Appended training summary to %s", results_csv)

   # -----------------------
   # Cleanup
   # -----------------------
   train_ds = val_ds = None  # drop strong refs before cleanup
   keras.backend.clear_session()
   gc.collect()
   logger.info("Cleaned up after training")
      
   return model, history
