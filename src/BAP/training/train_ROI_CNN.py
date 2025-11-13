"""End-to-end trainer for the ROI-specific CNN branch of the bone-age pipeline.

The module orchestrates ROI crop generation, TensorFlow dataset assembly, mixed
precision model construction, training and optional test evaluation, metric
logging, as well as persistence of both metrics and experiment summaries.
Everything required to reproduce an ROI-CNN experiment is encapsulated here so
callers only need to supply filesystem paths and a `ProjectConfig`.
"""

from typing import Tuple
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
from BAP.utils.dataset_loader import make_roi_dataset
from BAP.utils.path_manager import save_model_dicts

from BAP.roi.ROI_locator import train_locator_and_save_rois

from BAP.models.ROI_CNN import build_ROI_CNN

from BAP.training.callbacks import make_callbacks
from BAP.training.summary import append_summary_row

logger = get_logger(__name__)

def train_ROI_CNN(
   paths: dict,
   config_bundle: ProjectConfig, 
   save_dir: str
) -> Tuple[keras.Model, keras.callbacks.History]:
   """Train the ROI-CNN model, including ROI extraction, datasets, and reports.

   Parameters
   ----------
   paths:
      Mapping from split names (``"train"``, ``"validation"``, ``"test"``) to
      their raw DICOM/PNG directories. Used when ROIs must be re-generated.
   config_bundle:
      Fully-populated :class:`~BAP.utils.config.ProjectConfig` holding data,
      ROI, model, and training hyperparameters. These sub-configs drive ROI
      loader behavior, model architecture, optimizer setup, callbacks, and
      summary bookkeeping.
   save_dir:
      Destination directory where checkpoints, histories, and JSON summaries are
      written. The directory is expected to exist or be creatable by the caller.

   Returns
   -------
   model, history:
      The trained :class:`keras.Model` instance and its
      :class:`keras.callbacks.History`. They can be used for downstream
      inference or additional analysis.

   Workflow
   --------
   1. Mirror stdout for Keras logs, load configs, and enforce mixed precision.
   2. Ensure ROI crops exist (or call the locator to create them) for every
      split, timing the extraction process.
   3. Build TensorFlow datasets from ROI directories plus metadata CSVs, map
      them into the multi-input signature expected by the ROI-CNN, and enable
      pipelining via batching/prefetching.
   4. Instantiate the ROI-CNN with the configured channel counts, dense units,
      dropout, and optional gender branch; compile it with mixed-precision Adam,
      Huber loss, and regression metrics.
   5. Configure callbacks (checkpointing, early stopping, LR scheduling),
      launch training, and optionally evaluate on the test set.
   6. Record best-epoch metrics into a CSV summary, persist detailed histories
      as JSON, log a model summary, and clean up TensorFlow state to avoid GPU
      memory leaks.
   """

   mirror_keras_stdout_to_file()

   # -----------------------
   # Config & reproducibility
   # -----------------------
   model_name = "ROI_CNN"
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
   # ROI Datasets
   # -----------------------
   train_roi_dir = roi_paths["train"]
   val_roi_dir = roi_paths["validation"]
   
   train_metadata = pd.read_csv("data/metadata/train.csv")
   val_metadata = pd.read_csv("data/metadata/validation.csv")
   
   train_ds = make_roi_dataset(
      roi_dir=train_roi_dir,
      metadata=train_metadata,
   )
   
   val_ds = make_roi_dataset(
      roi_dir=val_roi_dir,
      metadata=val_metadata,
   )
   logger.info("Prepared training and validation ROI datasets.")


   batch_size = data_cfg.batch_size
   train_ds = train_ds.map(
      lambda features, age: ({"carpal": features["carpal"], "metaph": features["metaph"],
                              "gender": tf.cast(features["gender"], tf.int32)}, age),
      num_parallel_calls=tf.data.AUTOTUNE,
   )
   train_ds = train_ds.batch(batch_size)
   train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

   val_ds = val_ds.map(
      lambda features, age: ({"carpal": features["carpal"], "metaph": features["metaph"],
                              "gender": tf.cast(features["gender"], tf.int32)}, age),
      num_parallel_calls=tf.data.AUTOTUNE,
   )
   val_ds = val_ds.batch(batch_size)
   val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
   
   # Deduce ROI shape from one batch
   sample_batch = next(iter(train_ds.take(1)))
   roi_shape = tuple(sample_batch[0]["carpal"].shape[1:]) # [H, W, 1]
   
   # -----------------------
   # Model
   # -----------------------
   channels = model_cfg.roi_channels
   dense_units = model_cfg.roi_dense_units
   dropout_rate = model_cfg.dropout_rate
   use_gender = model_cfg.use_gender
   
   model = build_ROI_CNN(
      roi_shape=roi_shape,
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
      factor=0.25,
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
   logger.info("Starting ROI-CNN training for %d epochs", epochs)
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
   test_mae = float("nan")
   test_rmse = float("nan")
   if perform_test:
      test_roi_dir = roi_paths["test"]
      test_metadata = pd.read_csv("data/metadata/test.csv")
      logger.info("Evaluating %s on the test split.", model_name)
      test_ds = make_roi_dataset(
         roi_dir=test_roi_dir,
         metadata=test_metadata,
      )
      test_ds = test_ds.map(
         lambda features, age: ({"carpal": features["carpal"], "metaph": features["metaph"],
                                 "gender": tf.cast(features["gender"], tf.int32)}, age),
         num_parallel_calls=tf.data.AUTOTUNE,
      )
      test_ds = test_ds.batch(batch_size)
      test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
      test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
      test_mae = float(test_metrics.get("mae", float("nan")))
      test_rmse = float(test_metrics.get("rmse", float("nan")))
      logger.info(
         "Test metrics — MAE: %.4f, RMSE: %.4f",
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
   train_ds = val_ds = test_ds = None  # drop strong refs before cleanup
   keras.backend.clear_session()
   gc.collect()
   logger.info("Cleaned up after training")
      
   return model, history
