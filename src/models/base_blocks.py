from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Tensor

def conv_bn_relu(x: tf.Tensor, filters:int, k_size:int = 3, s:int = 1) -> tf.Tensor:
   """
   Applies Conv2D followed by BatchNormalization and ReLU activation.

   Args:
      x (tf.Tensor): Input feature map of shape [B, H, W, C].
      filters (int): Number of filters in the Conv2D layer.
      k_size (int, optional): Kernel size for the Conv2D layer. Defaults to 3.
      s (int, optional): Stride for the Conv2D layer. Defaults to 1.

   Returns:
      tf.Tensor: Output feature map of shape [B, H, W, filters].
   """
   x = layers.Conv2D(filters=filters, kernel_size=k_size, strides=s,
                     padding="same", use_bias=False)(x)
   x = layers.BatchNormalization()(x)
   x = layers.ReLU()(x)
   return x


