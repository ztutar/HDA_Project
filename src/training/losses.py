from typing import Literal
import tensorflow as tf

LossName = Literal["huber", "mse", "mae"]

def get_loss(loss_name: LossName = "huber", delta: float = 10.0) -> tf.keras.losses.Loss:
   name = name.lower()
   if name == "huber":
      return tf.keras.losses.Huber(delta=delta, naöe="huber")
   elif name == "mse":
      return tf.keras.losses.MeanSquaredError(name="mse")
   elif name == "mae":
      return tf.keras.losses.MeanAbsoluteError(name="mae")
   else:
      raise ValueError(f"Unsupported loss name: {loss_name}. Choose from 'huber', 'mse', or 'mae'.")