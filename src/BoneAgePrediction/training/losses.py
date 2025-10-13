
"""Utility helpers for choosing TensorFlow loss functions during model training.

This module exposes a `LossName` that captures the supported loss identifiers 
(`"huber"`, `"mse"`, and `"mae"`). The `get_loss` function maps a requested 
loss name to the corresponding `tf.keras.losses` implementation so that other 
training components can request standardized loss objects.
"""

from typing import Literal
import tensorflow as tf
import logging
try:
   from BoneAgePrediction.utils.logger import get_logger  
except ImportError:  # fallback when package not installed
   get_logger = logging.getLogger  

logger = get_logger(__name__)

LossName = Literal["huber", "mse", "mae"]

def get_loss(loss_name: LossName = "huber", delta: float = 10.0) -> tf.keras.losses.Loss:
   """Return a configured TensorFlow loss instance based on the requested name.

   Parameters:
      loss_name: Case-insensitive identifier of the desired loss. Accepts
         ``"huber"``, ``"mse"``, or ``"mae"`` and selects the corresponding
         `tf.keras.losses` implementation used during model compilation.
      delta: Non-negative transition point for the Huber loss that controls
         where the quadratic region switches to linear. The value is only
         applied when ``loss_name`` resolves to Huber; it is ignored for other
         loss selections.

   Returns:
      tf.keras.losses.Loss: Keras loss object ready to be passed into 
      `Model.compile`.

   Raises:
      ValueError: If an unsupported loss name is provided.
   """
   loss_key = loss_name.lower()
   logger.info("Building loss function '%s'", loss_key)
   if loss_key == "huber":
      return tf.keras.losses.Huber(delta=delta, name="huber")
   elif loss_key == "mse":
      return tf.keras.losses.MeanSquaredError(name="mse")
   elif loss_key == "mae":
      return tf.keras.losses.MeanAbsoluteError(name="mae")
   else:
      raise ValueError(f"Unsupported loss name: {loss_name}. Choose from 'huber', 'mse', or 'mae'.")
