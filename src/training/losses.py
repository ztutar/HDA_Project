
"""Utility helpers for choosing TensorFlow loss functions during model training.

This module exposes a `LossName` that captures the supported loss identifiers 
(`"huber"`, `"mse"`, and `"mae"`). The `get_loss` function maps a requested 
loss name to the corresponding `tf.keras.losses` implementation so that other 
training components can request standardized loss objects.
"""

from typing import Literal
import tensorflow as tf

LossName = Literal["huber", "mse", "mae"]

def get_loss(loss_name: LossName = "huber", delta: float = 10.0) -> tf.keras.losses.Loss:
   name = name.lower()
   if name == "huber":
      return tf.keras.losses.Huber(delta=delta, name="huber")
   elif name == "mse":
      return tf.keras.losses.MeanSquaredError(name="mse")
   elif name == "mae":
      return tf.keras.losses.MeanAbsoluteError(name="mae")
   else:
      raise ValueError(f"Unsupported loss name: {loss_name}. Choose from 'huber', 'mse', or 'mae'.")
