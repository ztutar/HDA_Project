
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

def estimate_gmacs(model: Model, input_shape: Tuple[int, int, int]) -> float:
   """
   Estimate total GMACs (Giga Multiply–Accumulate operations) per forward pass, 
   including Conv2D, Dense, BatchNorm, ReLU, and Pooling layers. 
   
   Args:
      model (keras.Model): The Keras model.
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
   logger.info("Initialized model %s for GMAC estimation using input_shape=%s", getattr(model, "name", "<unnamed>"), input_shape)

   def _tensor_shape(shape: tf.TensorShape):
      if shape is None:
         return None
      if isinstance(shape, list):
         shape = shape[0]
      return tf.TensorShape(shape)

   def _layer_output_shape(layer: layers.Layer):
      output_shape = getattr(layer, "output_shape", None)
      output_shape = _tensor_shape(output_shape)
      if output_shape is not None:
         return output_shape
      try:
         inferred = layer.compute_output_shape(getattr(layer, "input_shape", None))
      except Exception:
         return None
      return _tensor_shape(inferred)

   conv_layer_types = [layers.Conv2D]
   dense_layer_types = [layers.Dense]
   batchnorm_layer_types = [layers.BatchNormalization]
   activation_layer_types = [layers.ReLU, layers.Activation]
   pooling_layer_types = [
      layers.MaxPooling2D,
      layers.AveragePooling2D,
      layers.GlobalAveragePooling2D,
   ]
   
   model_layers = model.layers if hasattr(model, "layers") else None
   if model_layers is not None:
      for collection, class_name in (
         (conv_layer_types, "Conv2D"),
         (dense_layer_types, "Dense"),
         (batchnorm_layer_types, "BatchNormalization"),
         (activation_layer_types, "ReLU"),
         (activation_layer_types, "Activation"),
         (pooling_layer_types, "MaxPooling2D"),
         (pooling_layer_types, "AveragePooling2D"),
         (pooling_layer_types, "GlobalAveragePooling2D"),
      ):
         keras_cls = getattr(model_layers, class_name, None)
         if keras_cls is not None and keras_cls not in collection:
            collection.append(keras_cls)
            logger.info("Registered %s layer class from model namespace for GMACs accounting.", class_name)
   else:
      logger.info(
         "Model %s misses a 'layers' namespace; using default layer collections for GMACs accounting.",
         type(model).__name__,
      )

   conv_layer_types = tuple(conv_layer_types)
   dense_layer_types = tuple(dense_layer_types)
   batchnorm_layer_types = tuple(batchnorm_layer_types)
   activation_layer_types = tuple(activation_layer_types)
   pooling_layer_types = tuple(pooling_layer_types)

   conv_layer_names = {"Conv2D"}
   dense_layer_names = {"Dense"}
   batchnorm_layer_names = {"BatchNormalization"}
   activation_layer_names = {"ReLU", "Activation"}
   pooling_layer_names = {"MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D"}

   for layer in model.layers:
      layer_name = layer.__class__.__name__
      logger.info("Evaluating layer %s (%s) for GMACs estimation.", getattr(layer, "name", "<unnamed>"), layer_name)
      # --- Convolutional layers ---
      if isinstance(layer, conv_layer_types) or layer_name in conv_layer_names:
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            logger.info("Skipping conv layer %s: unable to determine output shape.", getattr(layer, "name", "<unnamed>"))
            continue
         H_out, W_out, C_out = int(output_shape[1]), int(output_shape[2]), int(output_shape[3])
         kernel = getattr(layer, "kernel", None)
         if kernel is None:
            logger.info("Skipping conv layer %s: kernel attribute missing.", getattr(layer, "name", "<unnamed>"))
            continue
         kernel_shape = kernel.shape
         K_h, K_w = int(kernel_shape[0]), int(kernel_shape[1])
         K_in = int(kernel_shape[2])
         conv_macs = H_out * W_out * C_out * (K_h * K_w * K_in)
         total_macs += conv_macs
         logger.info(
            "Conv layer %s: output_shape=%s kernel_shape=%s -> %.0f MACs",
            getattr(layer, "name", "<unnamed>"),
            tuple(output_shape),
            tuple(kernel_shape),
            conv_macs,
         )

      # --- Dense (Fully Connected) layers ---
      elif isinstance(layer, dense_layer_types) or layer_name in dense_layer_names:
         kernel = getattr(layer, "kernel", None)
         if kernel is None:
            logger.info("Skipping dense layer %s: kernel attribute missing.", getattr(layer, "name", "<unnamed>"))
            continue
         kernel_shape = kernel.shape
         units_in = int(kernel_shape[0])
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            logger.info("Skipping dense layer %s: unable to determine output shape.", getattr(layer, "name", "<unnamed>"))
            continue
         units_out = int(kernel_shape[1] if len(kernel_shape) > 1 else output_shape[-1])
         dense_macs = units_in * units_out
         total_macs += dense_macs
         logger.info(
            "Dense layer %s: units_in=%d units_out=%d -> %.0f MACs",
            getattr(layer, "name", "<unnamed>"),
            units_in,
            units_out,
            dense_macs,
         )

      # --- Batch Normalization ---
      elif isinstance(layer, batchnorm_layer_types) or layer_name in batchnorm_layer_names:
         # 2 MACs per output element (normalize and scale)
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            logger.info("Skipping batchnorm layer %s: unable to determine output shape.", getattr(layer, "name", "<unnamed>"))
            continue
         elements = np.prod(output_shape[1:])  # exclude batch dimension
         bn_macs = 2 * elements
         total_macs += bn_macs
         logger.info(
            "BatchNorm layer %s: output_shape=%s -> %.0f MACs",
            getattr(layer, "name", "<unnamed>"),
            tuple(output_shape),
            bn_macs,
         )

      # --- Activation functions (ReLU, etc.) ---
      elif isinstance(layer, activation_layer_types) or layer_name in activation_layer_names:
         # 1 MAC per output element (comparison)
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            logger.info("Skipping activation layer %s: unable to determine output shape.", getattr(layer, "name", "<unnamed>"))
            continue
         elements = np.prod(output_shape[1:])  # exclude batch dimension
         activation_macs = elements
         total_macs += activation_macs
         logger.info(
            "Activation layer %s: output_shape=%s -> %.0f MACs",
            getattr(layer, "name", "<unnamed>"),
            tuple(output_shape),
            activation_macs,
         )

      # --- Pooling layers (Max/Avg) ---
      elif isinstance(layer, pooling_layer_types) or layer_name in pooling_layer_names:
         # 1 MAC per output element (sum/avg)
         output_shape = _layer_output_shape(layer)
         if output_shape is None:
            logger.info("Skipping pooling layer %s: unable to determine output shape.", getattr(layer, "name", "<unnamed>"))
            continue
         elements = np.prod(output_shape[1:])  # exclude batch dimension
         pooling_macs = elements 
         total_macs += pooling_macs
         logger.info(
            "Pooling layer %s: output_shape=%s -> %.0f MACs",
            getattr(layer, "name", "<unnamed>"),
            tuple(output_shape),
            pooling_macs,
         )
      
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
      
   def on_train_begin(self) -> None:
      self.epoch_times = []
      logger.info("EpochTimer started timing training run.")
   
   def on_epoch_end(self, epoch: int) -> None:
      self.epoch_times.append(time.time() - self.t0)
      duration = self.epoch_times[-1]
      logger.info("Epoch %d duration: %.2f sec", epoch + 1, duration)
