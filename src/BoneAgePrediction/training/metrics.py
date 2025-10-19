
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
import logging
try:
   from BoneAgePrediction.utils.logger import get_logger  
except ImportError:  # fallback when package not installed
   get_logger = logging.getLogger  

logger = get_logger(__name__)

def mae() -> tf.keras.metrics.Metric:
   """
   Mean Absolute Error metric for regression tasks.
   
   Returns:
      tf.keras.metrics.Metric: Tracks MAE over batches.
   """
   return tf.keras.metrics.MeanAbsoluteError(name="mae")

def rmse() -> tf.keras.metrics.Metric:
   """
   Root Mean Squared Error metric for regression tasks.
   
   Returns:
      tf.keras.metrics.Metric: Tracks RMSE over batches.
   """
   return tf.keras.metrics.RootMeanSquaredError(name="rmse")

def count_params(model: tf.keras.Model) -> int:
   """
   Counts the total number of trainable + non-trainable parameters in a Keras model.
   
   Args:
      model (tf.keras.Model): The Keras model.
   Returns:
      int: Total number of parameters.
   """
   total = int(model.count_params())
   logger.debug("Model %s has %d parameters", getattr(model, "name", "<unnamed>"), total)
   return total

def estimate_gmacs(model: tf.keras.Model, input_shape: Tuple[int, int, int]) -> float:
   """
   Estimate total GMACs (Giga Multiply–Accumulate operations) per forward pass, 
   including Conv2D, Dense, BatchNorm, ReLU, and Pooling layers. 
   
   Args:
      model (tf.keras.Model): The Keras model.
      input_shape (Tuple[int, int, int]): Input tensor shape [H, W, C].
   
   Returns:
      float: Estimated GMacs for a single forward pass.
   Notes:
      - Each multiply or add counts as 1 MAC.
      - BatchNorm/ReLU/Pooling operations are approximated as 1 MAC per output element.
   """
   if not model.built:
      model.build((None, *input_shape))
   
   total_macs = 0.0
   
   # Run one dummy forward pass to initialize layer shapes (so input/output_shape are defined for MACs).
   def _random_like(tensor_shape: tf.TensorShape) -> tf.Tensor:
      dims = [1 if dim is None else int(dim) for dim in tensor_shape]
      if not dims:
         dims = [1]
      return tf.random.uniform((1, *dims), dtype=tf.float32)

   def _tensor_key(tensor: tf.Tensor, index: int) -> str:
      name = getattr(tensor, "name", None)
      if not name:
         return f"input_{index}"
      return name.split(":")[0]

   input_tensors = model.inputs if isinstance(model.inputs, (list, tuple)) else [model.inputs]
   if len(input_tensors) > 1:
      model_feed = {}
      for idx, tensor in enumerate(input_tensors):
         key = _tensor_key(tensor, idx)
         model_feed[key] = _random_like(tf.TensorShape(tensor.shape)[1:])
   else:
      model_feed = _random_like(tf.TensorShape(input_tensors[0].shape)[1:])
   _ = model(model_feed, training=False)
   logger.debug("Initialized model %s for GMAC estimation using input_shape=%s", getattr(model, "name", "<unnamed>"), input_shape)

   def _tensor_shape(shape):
      if shape is None:
         return None
      if isinstance(shape, list):
         shape = shape[0]
      return tf.TensorShape(shape)

   def _layer_output_shape(layer):
      output_shape = getattr(layer, "output_shape", None)
      output_shape = _tensor_shape(output_shape)
      if output_shape is not None:
         return output_shape
      try:
         inferred = layer.compute_output_shape(getattr(layer, "input_shape", None))
      except Exception:
         return None
      return _tensor_shape(inferred)

   for layer in model.layers:
      # --- Convolutional layers ---
      if isinstance(layer, tf.keras.layers.Conv2D):
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            continue
         H_out, W_out, C_out = int(output_shape[1]), int(output_shape[2]), int(output_shape[3])
         kernel = getattr(layer, "kernel", None)
         if kernel is None:
            continue
         kernel_shape = kernel.shape
         K_h, K_w = int(kernel_shape[0]), int(kernel_shape[1])
         K_in = int(kernel_shape[2])
         conv_macs = H_out * W_out * C_out * (K_h * K_w * K_in)
         total_macs += conv_macs

      # --- Dense (Fully Connected) layers ---
      elif isinstance(layer, tf.keras.layers.Dense):
         kernel = getattr(layer, "kernel", None)
         if kernel is None:
            continue
         kernel_shape = kernel.shape
         units_in = int(kernel_shape[0])
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            continue
         units_out = int(kernel_shape[1] if len(kernel_shape) > 1 else output_shape[-1])
         dense_macs = units_in * units_out
         total_macs += dense_macs

      # --- Batch Normalization ---
      elif isinstance(layer, tf.keras.layers.BatchNormalization):
         # 2 MACs per output element (normalize and scale)
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            continue
         elements = np.prod(output_shape[1:])  # exclude batch dimension
         bn_macs = 2 * elements
         total_macs += bn_macs

      # --- Activation functions (ReLU, etc.) ---
      elif isinstance(layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.Activation):
         # 1 MAC per output element (comparison)
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            continue
         elements = np.prod(output_shape[1:])  # exclude batch dimension
         activation_macs = elements
         total_macs += activation_macs

      # --- Pooling layers (Max/Avg) ---
      elif isinstance(layer, (tf.keras.layers.MaxPooling2D, 
                              tf.keras.layers.AveragePooling2D, 
                              tf.keras.layers.GlobalAveragePooling2D)):
         # 1 MAC per output element (sum/avg)
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            continue
         elements = np.prod(output_shape[1:])  # exclude batch dimension
         pooling_macs = elements 
         total_macs += pooling_macs
      
      # --- Other layers can be added here as needed ---
      else:
         pass  # Ignore other layers for MACs estimation
      
   gmacs = total_macs / 1e9  # Convert to GMacs
   logger.info("Estimated %.4f GMACs for model %s", gmacs, getattr(model, "name", "<unnamed>"))
   return gmacs


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
      
   def on_epoch_begin(self, epoch: int, logs=None) -> None:
      self.t0 = time.time()
      logger.debug("Epoch %d started.", epoch + 1)
   
   def on_epoch_end(self, epoch: int, logs=None) -> None:
      self.epoch_times.append(time.time() - self.t0)
      duration = self.epoch_times[-1]
      logger.info("Epoch %d duration: %.2f sec", epoch + 1, duration)
