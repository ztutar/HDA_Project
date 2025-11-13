#TODO: add docstring explanation for this module. Write in details and explain each function and class.

from typing import Tuple
from dataclasses import asdict
import os
import gc
import io
import glob
import numpy as np
import tensorflow as tf
import keras
import time
import pandas as pd
from pathlib import Path

from BAP.utils.logger import get_logger, mirror_keras_stdout_to_file
from BAP.utils.config import ProjectConfig
from BAP.utils.dataset_loader import make_fusion_dataset
from BAP.utils.path_manager import save_model_dicts


from BAP.roi.ROI_locator import train_locator_and_save_rois

from BAP.models.Fusion_CNN import build_FusionCNN

from BAP.training.callbacks import make_callbacks
from BAP.training.summary import append_summary_row

logger = get_logger(__name__)

def train_FusionCNN(
   paths: dict,
   config_bundle: ProjectConfig,
   save_dir: str
) -> Tuple[keras.Model, keras.callbacks.History]:
   
   #TODO: add docstring explanation for this function. write in details and explain each step
   """
   Train the Fusion CNN model end to end on full-hand images plus ROI crops.
   """
   mirror_keras_stdout_to_file()

   # -----------------------
   # Config & reproducibility
   # -----------------------
   model_name = "FusionCNN"
   data_cfg = config_bundle.data
   roi_cfg = config_bundle.roi
   locator_cfg = roi_cfg.locator

   model_cfg = config_bundle.model
   training_cfg = config_bundle.training

   policy_name = "mixed_float16"
   if keras.mixed_precision.global_policy().name != policy_name:
      keras.mixed_precision.set_global_policy(policy_name)
      logger.info("Set mixed-precision policy to %s", policy_name)

   # -----------------------
   # ROI Locator & Extractor
   # -----------------------
   roi_path = locator_cfg.roi_path
   os.makedirs(roi_path, exist_ok=True)

   def _has_pngs(path: str) -> bool:
      return os.path.isdir(path) and glob.glob(os.path.join(path, "*.png"))

   # Generate crops for each split (train/val/test) if not already present
   roi_extraction_time = 0.0
   roi_paths = {}
   for split in ["train", "validation", "test"]:
      carpal_dir = os.path.join(roi_path, split, "carpal")
      metaph_dir = os.path.join(roi_path, split, "metaph")
      heatmaps_dir = os.path.join(roi_path, split, "heatmaps")
      roi_paths[split] = {
         "carpal": Path(carpal_dir), 
         "metaph": Path(metaph_dir), 
         "heatmaps": Path(heatmaps_dir)
         }
      if not (_has_pngs(carpal_dir) and _has_pngs(metaph_dir)):
         logger.info("Generating ROIs for %s split into %s.", split, roi_path)
         roi_time_start = time.time()
         train_locator_and_save_rois(
            data_path=paths[split], 
            roi_paths=roi_paths[split],
            config=config_bundle, 
            split=split
         )
         roi_time_end = time.time()
         roi_extraction_time += roi_time_end - roi_time_start
   logger.info("Total ROI extraction time: %.2fs", roi_extraction_time)

   # -----------------------
   # Fusion Dataset
   # -----------------------
   train_image_dir = Path(paths["train"])
   val_image_dir = Path(paths["val"])
   
   train_roi_dir = roi_paths["train"]
   val_roi_dir = roi_paths["validation"]
   
   train_metadata = pd.read_csv("data/metadata/train.csv")
   val_metadata = pd.read_csv("data/metadata/validation.csv")
   
   
   image_size = data_cfg.image_size
   clahe = data_cfg.clahe
   augment = data_cfg.augment
   
   train_ds = make_fusion_dataset(
      image_dir=train_image_dir,
      roi_dir=train_roi_dir,
      metadata=train_metadata,
      image_size=image_size,
      clahe=clahe,
      augment=augment,
   )

   val_ds = make_fusion_dataset(
      image_dir=val_image_dir,
      roi_dir=val_roi_dir,
      metadata=val_metadata,
      image_size=image_size,
      clahe=clahe,
      augment=False,
   )
   logger.info("Prepared fusion training and validation datasets from %s", data_cfg.data_path)

   batch_size = data_cfg.batch_size
   train_ds = train_ds.map(
      lambda features, age: ({"image": features["image"],
                              "carpal": features["carpal"], "metaph": features["metaph"],
                              "gender": tf.cast(features["gender"], tf.int32)}, age),
      num_parallel_calls=tf.data.AUTOTUNE,
   )
   train_ds = train_ds.batch(batch_size)
   train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
   
   val_ds = val_ds.map(
      lambda features, age: ({"image": features["image"],
                              "carpal": features["carpal"], "metaph": features["metaph"],
                              "gender": tf.cast(features["gender"], tf.int32)}, age),
      num_parallel_calls=tf.data.AUTOTUNE,
   )
   val_ds = val_ds.batch(batch_size)
   val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

   # Deduce input shapes from one batch
   sample_inputs, _ = next(iter(train_ds.take(1)))
   global_input_shape = tuple(sample_inputs["image"].shape[1:])
   roi_shape = tuple(sample_inputs["carpal"].shape[1:])

   # -----------------------
   # Model
   # -----------------------
   global_channels = model_cfg.global_channels
   global_dense_units = model_cfg.global_dense_units
   roi_channels = model_cfg.roi_channels
   roi_dense_units = model_cfg.roi_dense_units
   fusion_dense_units = model_cfg.fusion_dense_units
   dropout_rate = model_cfg.dropout_rate
   use_gender = model_cfg.use_gender


   model = build_FusionCNN(
      global_input_shape=global_input_shape,
      roi_shape=roi_shape,
      global_channels=global_channels,
      roi_channels=roi_channels,
      global_dense_units=global_dense_units,
      roi_dense_units=roi_dense_units,
      fusion_dense_units=fusion_dense_units,
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
      metrics=metrics
   )
   logger.info("Model compiled with Huber loss and Adam optimizer (LR=%.6f)", learning_rate)

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
      factor=0.2,
      patience=4, 
      min_lr=1e-6, 
      verbose=1
   )
   callbacks.append(reduce_lr)
   logger.info("Configured training callbacks with patience: %d", patience)

   # -----------------------
   # Train
   # -----------------------
   epochs = training_cfg.epochs
   logger.info("Starting FusionCNN training for %d epochs", epochs)
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
   training_time = end_train - start_train + roi_extraction_time
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
   perform_test = training_cfg.perform_test
   test_loss = float("nan")
   test_mae = float("nan")
   test_rmse = float("nan")
   if perform_test:
      test_image_dir = Path(paths["test"])
      test_roi_dir = roi_paths["test"]
      test_metadata = pd.read_csv("data/metadata/test.csv")
      logger.info("Evaluating %s on the test split.", model_name)
      test_ds = make_fusion_dataset(
         image_dir=test_image_dir,
         roi_dir=test_roi_dir,
         metadata=test_metadata,
         image_size=image_size,
         clahe=clahe,
         augment=False,
   )
      test_ds = test_ds.map(
         lambda features, age: ({"image": features["image"],
                                 "carpal": features["carpal"], "metaph": features["metaph"],
                                 "gender": tf.cast(features["gender"], tf.int32)}, age),
         num_parallel_calls=tf.data.AUTOTUNE,
      )
      test_ds = test_ds.batch(batch_size)
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

   train_mae = history.history.get("mae", float("nan"))[best_epoch_idx]
   train_rmse = history.history.get("rmse", float("nan"))[best_epoch_idx]
   val_mae = history.history.get("val_mae", float("nan"))[best_epoch_idx]
   val_rmse = history.history.get("val_rmse", float("nan"))[best_epoch_idx]

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
      "total_training_time_s": total_time_display,
      "train_mae": f"{train_mae:.4f}",
      "train_rmse": f"{train_rmse:.4f}",
      "val_mae": f"{val_mae:.4f}",
      "val_rmse": f"{val_rmse:.4f}",
      "test_mae": f"{test_mae:.4f}",
      "test_rmse": f"{test_rmse:.4f}",
      "stopped_epoch": num_epochs_ran,
      "best_epoch": best_epoch_idx+1,
      "save_dir": save_dir,
   }
   append_summary_row(
      results_csv=results_csv,
      base_data=summary_base,
      config_bundle=config_bundle,
   )
   logger.info("Appended training summary to %s", results_csv)

   # -----------------------
   # Save model results & metrics
   # -----------------------
   model_results_dict = {
      "num_params": num_params,
      "training_time": training_time,
      "num_epochs_ran": num_epochs_ran,
      "best_epoch_idx": best_epoch_idx
   }
   model_metrics_dict = {
      "history": history.history,
      "train_loss": history.history["loss"][best_epoch_idx],
      "train_mae": history.history["mae"][best_epoch_idx],
      "train_rmse": history.history["rmse"][best_epoch_idx],
      "val_loss": history.history["val_loss"][best_epoch_idx],
      "val_mae": history.history["val_mae"][best_epoch_idx],
      "val_rmse": history.history["val_rmse"][best_epoch_idx],
   }
   if perform_test:
      model_metrics_dict.update({
         "test_loss": test_metrics,
         "test_mae": test_mae,
         "test_rmse": test_rmse
      })
   save_model_dicts(model_results_dict, os.path.join(save_dir, "model_results.json"))
   save_model_dicts(model_metrics_dict, os.path.join(save_dir, "model_metrics.json"))

   # -----------------------
   # Cleanup
   # -----------------------
   train_ds = val_ds = test_ds = None
   keras.backend.clear_session()
   gc.collect()
   logger.info("Cleaned up session after training")

   return model, history
