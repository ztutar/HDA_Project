
"""Utility module with helpers for monitoring Keras training runs.

It exposes functions `mae` and `rmse`, which return ready-to-use
regression metrics that keep track of prediction error over incoming batches.
The `count_params` helper reports how many parameters a model owns and
`estimate_gmacs` approximates how many multiply-accumulate operations a single
forward pass consumes. The `EpochTimer` callback records the duration of each
training epoch so that logs and reports can include timing information.
"""


from typing import Dict, Tuple
import time
import numpy as np
import tensorflow as tf
from keras import layers, metrics, Model  

from BoneAgePrediction.utils.logger import get_logger  

logger = get_logger(__name__)

def mae() -> metrics.Metric:
   """
   Mean Absolute Error metric for regression tasks.
   
   Returns:
      keras.metrics.Metric: Tracks MAE over batches.
   """
   return metrics.MeanAbsoluteError(name="mae")

def rmse() -> metrics.Metric:
   """
   Root Mean Squared Error metric for regression tasks.
   
   Returns:
      keras.metrics.Metric: Tracks RMSE over batches.
   """
   return metrics.RootMeanSquaredError(name="rmse")

def count_params(model: Model) -> int:
   """
   Counts the total number of trainable + non-trainable parameters in a Keras model.
   
   Args:
      model (keras.Model): The Keras model.
   Returns:
      int: Total number of parameters.
   """
   total = int(model.count_params())
   logger.info("Model %s has %d parameters", getattr(model, "name", "<unnamed>"), total)
   return total


class EpochTimer(tf.keras.callbacks.Callback):
   """
   Keras Callback to measure and log the duration of each training epoch.
   
   Usage:
      timer = EpochTimer()
      model.fit(..., callbacks=[timer])
   """
   def __init__(self) -> None:
      super().__init__()
      self.epoch_times: list[float] = []
      self.t0 = 0.0
      
   def on_train_begin(self, logs=None) -> None:
      self.epoch_times = []
      logger.info("EpochTimer started timing training run.")
   
   def on_epoch_end(self, epoch: int, logs=None) -> None:
      self.epoch_times.append(time.time() - self.t0)
      duration = self.epoch_times[-1]
      logger.info("Epoch %d duration: %.2f sec", epoch + 1, duration)
