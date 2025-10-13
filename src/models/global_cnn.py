
"""
This module defines a configurable convolutional neural network that produces a
single numeric prediction from input images. The network is built out of
stacked convolutional blocks that perform feature extraction, followed by a
global pooling layer that condenses the spatial information into one vector,
and a pair of dense layers that map the extracted features to the final output.

The `build_global_cnn` function is the main entry point. It lets callers
select the input shape, how many convolutional blocks to use, how many filters
each block should contain, and the size of the dense layer before the output.
Internally it relies on the shared `conv_bn_relu` helper from
`src.models.base_blocks` to assemble each convolutional block, creating a model
that can be reused for different experiments with minimal changes.
"""


from typing import Sequence, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model, Tensor
from src.models.base_blocks import conv_bn_relu

def build_global_cnn(input_shape: Tuple[int, int, int] = (512, 512, 1),
                     num_blocks: int = 3,
                     channels: Sequence[int] = (32, 64, 128),
                     dense_units: int = 64,
                     name: str = "B01_Global_CNN") -> tf.keras.Model:
   """
   Builds a global CNN model for feature extraction.
   
   Architecture:
      [ (Conv-BN-ReLU) x2 -> MaxPool ] * num_blocks
      -> GlobalAveragePooling -> Dense(dense_units) -> Dense(1)
   
   Args:
      input_shape (Tuple[int, int, int], optional): Shape of the input image. Defaults to (512, 512, 1).
      num_blocks (int, optional): Number of downsampling blocks (must equal len(channels)).Defaults to 3.
      channels (Sequence[int], optional): Number of channels for each block. Length should be equal to num_blocks. Defaults to (32, 64, 128).
      dense_units (int, optional): Number of units in the dense layer before the output. Defaults to 64.
      name (str, optional): Name of the Keras model. Defaults to "B01_Global_CNN".
   
   Returns:
      tf.keras.Model: Compiled model mapping image -> age (months).
                     Inputs:  image: float32 [B,H,W,1]
                     Outputs: age  : float32 [B,1]
   """
   assert num_blocks == len(channels), "num_blocks must equal len(channels)"
   input_image = layers.Input(shape=input_shape, dtype=tf.float32, name="image")
   
   x = input_image
   for ch in channels:
      # two conv layers per block, then downsample
      x = conv_bn_relu(x, filters=ch, k_size=3, s=1)
      x = conv_bn_relu(x, filters=ch, k_size=3, s=1)
      x = layers.MaxPooling2D(pool_size=2)(x)
      
   # global average pool, then dense layers
   x = layers.GlobalAveragePooling2D()(x)
   x = layers.Dense(units=dense_units, activation='relu')(x)
   output_age = layers.Dense(units=1, activation='linear', name='age_months')(x)
   
   model = Model(inputs=input_image, outputs=output_age, name=name)
   return model
