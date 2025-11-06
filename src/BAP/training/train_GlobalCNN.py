#TODO: add docstring explanation for this module. write in details and explain each function


from typing import Tuple
from dataclasses import asdict
import gc
import io
import numpy as np
import tensorflow as tf
import keras
import time

from BAP.utils.logger import get_logger, mirror_keras_stdout_to_file
from BAP.utils.config import ProjectConfig
from BAP.utils.dataset_loader import make_dataset

from BAP.models.Global_CNN import build_GlobalCNN

from BAP.training.callbacks import make_callbacks
from BAP.training.summary import append_summary_row

logger = get_logger(__name__)

def train_GlobalCNN(
   config_bundle: ProjectConfig, 
   save_dir: str
) -> Tuple[keras.Model, keras.callbacks.History]:
   #TODO: add docstring explanation for this function. write in details and explain each step

   mirror_keras_stdout_to_file()
   
   # -----------------------
   # Config & reproducibility
   # -----------------------
   model_name = "GlobalCNN"
   data_cfg = config_bundle.data
   model_cfg = config_bundle.model
   training_cfg = config_bundle.training
   
   config_dict = asdict(config_bundle)
   logger.info("Configuration parameters: %s", config_dict)

   policy_name = "mixed_float16" 
   if keras.mixed_precision.global_policy().name != policy_name:
      keras.mixed_precision.set_global_policy(policy_name)
      logger.info("Set mixed-precision policy to %s", policy_name)


   # -----------------------
   # Data
   # -----------------------
   data_path = data_cfg.data_path
   image_size = data_cfg.image_size
   keep_aspect_ratio = data_cfg.keep_aspect_ratio
   batch_size = data_cfg.batch_size
   clahe = data_cfg.clahe
   augment = data_cfg.augment
   
   train_ds = make_dataset(
      data_path=data_path,
      split='train',
      image_size=image_size,
      keep_aspect_ratio=keep_aspect_ratio,
      batch_size=batch_size,
      clahe=clahe,
      augment=augment,
   )
   
   val_ds = make_dataset(
      data_path=data_path,
      split='val',
      image_size=image_size,
      keep_aspect_ratio=keep_aspect_ratio,
      batch_size=batch_size,
      clahe=clahe,
      augment=False,
   )
   logger.info("Prepared training and validation datasets from %s", data_path)

   use_gender = model_cfg.use_gender
   def _select_inputs(features: dict, label: tf.Tensor):
      if use_gender:
         inputs = {
               "image": features["image"],
               "gender": tf.cast(features["gender"], tf.int32),
         }
         return inputs, label
      return features["image"], label


   train_ds = train_ds.map(_select_inputs, num_parallel_calls=tf.data.AUTOTUNE)
   train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

   val_ds = val_ds.map(_select_inputs, num_parallel_calls=tf.data.AUTOTUNE)
   val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
   
   # -----------------------
   # Model
   # -----------------------
   channels = model_cfg.global_channels
   dense_units = model_cfg.global_dense_units
   dropout_rate = model_cfg.dropout_rate
   input_shape = (image_size, image_size, 1)  # grayscale input
   
   model = build_GlobalCNN(
      input_shape=input_shape,
      channels=channels,
      dense_units=dense_units,
      dropout_rate=dropout_rate,
      use_gender=use_gender,
   )
   
   # -----------------------
   # Optimizer (Adam, fixed LR)
   # -----------------------
   learning_rate = training_cfg.learning_rate
   optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
   optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
   
   # -----------------------
   # Loss & Metrics
   # -----------------------
   loss_name = training_cfg.loss
   loss_fn = keras.losses.Huber(delta=10, name="huber")
   metrics = [
      keras.metrics.MeanAbsoluteError(name="mae"), 
      keras.metrics.RootMeanSquaredError(name="rmse")
   ]
   
   # -----------------------
   # Compile model
   # -----------------------
   model.compile(
      optimizer=optimizer,
      loss=loss_fn,
      metrics=metrics,
   )
   logger.info("Model compiled with %s loss and Adam optimizer (LR=%.6f)", loss_name, learning_rate)
   
   # -----------------------
   # Callbacks
   # -----------------------
   patience = training_cfg.patience
   callbacks = make_callbacks(
      save_dir=save_dir,
      model_name=model_name,
      patience=patience,
   )
   # learning rate schedule
   reduce_lr = keras.callbacks.ReduceLROnPlateau(
      monitor='val_mae', 
      factor=0.1,
      patience=4, 
      min_lr=1e-5, 
      verbose=1
   )
   callbacks.append(reduce_lr)
   logger.info("Configured training callbacks with patience: %d", patience)
   
   # -----------------------
   # Train
   # -----------------------
   epochs = training_cfg.epochs
   logger.info("Starting GlobalCNN training for %d epochs", epochs)
   start_train = time.time()
   history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=callbacks,
      verbose=1,
   )
   logger.info("Training finished")
   end_train = time.time()
   
   # -----------------------
   # Complexity & Timing
   # -----------------------
   num_params = int(model.count_params())
   training_time = end_train - start_train
   num_epochs_ran = len(history.history.get("loss", []))
   total_time_display = f"{training_time:.2f}" if training_time is not None else "n/a"
   logger.info(
      "[%s] Params: %d | Number of epochs ran: %d | Total training time: %ss",
      model_name,
      num_params,
      num_epochs_ran,
      total_time_display,
   )
   
   summary_stream = io.StringIO()
   model.summary(print_fn=lambda line: summary_stream.write(line + "\n"))
   logger.info("Model summary:\n%s", summary_stream.getvalue())
   
   # -----------------------
   # Test Evaluation (optional)
   # -----------------------
   perform_test = getattr(training_cfg, "perform_test", False)
   test_loss = float("nan")
   test_mae = float("nan")
   test_rmse = float("nan")
   if perform_test:
      logger.info("Evaluating %s on the test split.", model_name)
      test_ds = make_dataset(
         data_path=data_path,
         split="test",
         image_size=image_size,
         keep_aspect_ratio=keep_aspect_ratio,
         batch_size=batch_size,
         clahe=clahe,
         augment=False,
      )
      test_ds = test_ds.map(_select_inputs, num_parallel_calls=tf.data.AUTOTUNE)
      test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
      test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
      test_loss = float(test_metrics.get("loss", float("nan")))
      test_mae = float(test_metrics.get("mae", float("nan")))
      test_rmse = float(test_metrics.get("rmse", float("nan")))
      logger.info(
         "Test metrics — loss: %.4f, MAE: %.4f, RMSE: %.4f",
         test_loss,
         test_mae,
         test_rmse,
      )
   else:
      logger.info(
         "Skipping test evaluation because training.perform_test is %s.",
         perform_test,
      )
      test_ds = None
   
   # -----------------------
   # Summary CSV
   # -----------------------
   results_csv = training_cfg.results_csv
   val_mae_history = history.history.get("val_mae", [])
   best_epoch_idx = int(np.argmin(val_mae_history)) if val_mae_history else None

   def _metric_at(name: str, idx: int, default: float = float("nan")) -> float:
      series = history.history.get(name)
      if series is None or idx is None or idx >= len(series):
         return default
      return float(series[idx])

   train_mae = _metric_at("mae", best_epoch_idx)
   train_rmse = _metric_at("rmse", best_epoch_idx)
   val_mae = _metric_at("val_mae", best_epoch_idx)
   val_rmse = _metric_at("val_rmse", best_epoch_idx)

   logger.info(
      "Best epoch metrics — train MAE: %.4f, train RMSE: %.4f, val MAE: %.4f, val RMSE: %.4f",
      train_mae,
      train_rmse,
      val_mae,
      val_rmse,
   )

   early_stop_cb = next((cb for cb in callbacks if isinstance(cb, keras.callbacks.EarlyStopping)), None)
   if early_stop_cb is not None:
      stopped_epoch = int(getattr(early_stop_cb, "stopped_epoch", 0) or 0)

   summary_base = {
      "model_name": model_name,
      "num_params": num_params,
      "total_training_time_s": total_time_display,
      "train_mae": f"{train_mae:.4f}",
      "train_rmse": f"{train_rmse:.4f}",
      "val_mae": f"{val_mae:.4f}",
      "val_rmse": f"{val_rmse:.4f}",
      "test_mae": f"{test_mae:.4f}",
      "test_rmse": f"{test_rmse:.4f}",
      "stopped_epoch": stopped_epoch,
      "best_epoch": best_epoch_idx,
      "save_dir": save_dir,
   }
   append_summary_row(
      results_csv=results_csv,
      base_data=summary_base,
      config_bundle=config_bundle,
   )
   logger.info("Appended training summary to %s", results_csv)

   # -----------------------
   # Cleanup
   # -----------------------
   train_ds = val_ds = test_ds = None  # drop strong refs before cleanup
   keras.backend.clear_session()
   gc.collect()
   logger.info("Cleaned up session after training.")
      
   return model, history
