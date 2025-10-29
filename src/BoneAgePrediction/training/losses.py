
"""Utility helpers for choosing TensorFlow loss functions during model training.

This module exposes a `LossName` that captures the supported loss identifiers 
(`"huber"`, `"mse"`, and `"mae"`). The `get_loss` function maps a requested 
loss name to the corresponding `keras.losses` implementation so that other 
training components can request standardized loss objects.
"""

from typing import Literal
from keras import losses
from BoneAgePrediction.utils.logger import get_logger  

logger = get_logger(__name__)

LossName = Literal["huber", "mse", "mae"]

def get_loss(loss_name: LossName = "huber", huber_delta: float = 10.0) -> losses.Loss:
   """Return a configured TensorFlow loss instance based on the requested name.

   Parameters:
      loss_name: Case-insensitive identifier of the desired loss. Accepts
         ``"huber"``, ``"mse"``, or ``"mae"`` and selects the corresponding
         `keras.losses` implementation used during model compilation.
      delta: Non-negative transition point for the Huber loss that controls
         where the quadratic region switches to linear. The value is only
         applied when ``loss_name`` resolves to Huber; it is ignored for other
         loss selections.

   Returns:
      keras.losses.Loss: Keras loss object ready to be passed into 
      `Model.compile`.

   Raises:
      ValueError: If an unsupported loss name is provided.
   """
   loss_key = loss_name.lower()
   logger.info("Building loss function '%s'", loss_key)
   if loss_key == "huber":
      return losses.Huber(delta=huber_delta, name="huber")
   elif loss_key == "mse":
      return losses.MeanSquaredError(name="mse")
   elif loss_key == "mae":
      return losses.MeanAbsoluteError(name="mae")
   else:
      raise ValueError(f"Unsupported loss name: {loss_name}. Choose from 'huber', 'mse', or 'mae'.")
