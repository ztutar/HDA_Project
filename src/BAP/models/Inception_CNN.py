"""
This module provides functionality to build an Inception Convolutional Neural Network (CNN) model
for bone age prediction. The model is based on the Inception-v4 architecture and can optionally
include gender information as an additional input to predict age in months.
"""

from typing import Optional, Tuple

import tensorflow as tf
from keras import Model, layers


# Main model construction
def build_InceptionCNN(
   input_shape: Tuple[int, int, int] = (512, 512, 1),
   base_filters: int = 32,
   num_a_blocks: int = 4,
   num_b_blocks: int = 7,
   num_c_blocks: int = 3,
   head_dense_units: int = 128,
   head_dropout: float = 0.3,
   use_gender: bool = False,
) -> Model:
   """
   Builds an Inception CNN model for bone age prediction based on Inception-v4 architecture.
   
   Args:
      input_shape (Tuple[int, int, int]): Shape of the input image (height, width, channels). Default is (512, 512, 1).
      base_filters (int): Base number of filters used in the model. Default is 32.
      num_a_blocks (int): Number of Inception-A blocks. Default is 4.
      num_b_blocks (int): Number of Inception-B blocks. Default is 7.
      num_c_blocks (int): Number of Inception-C blocks. Default is 3.
      head_dense_units (int): Number of units in the head dense layer. Default is 128.
      head_dropout (float): Dropout rate for the head. Default is 0.3.
      use_gender (bool): Whether to include gender as an additional input. Default is False.
   
   Returns:
      Model: A Keras Model instance for bone age prediction.
   """
   image_input = layers.Input(shape=input_shape, dtype=tf.float32, name="image")

   x = Stem(image_input, base_filters=base_filters)

   for idx in range(num_a_blocks):
      x = Inception_A(x, base_filters=base_filters, name=f"inception_a{idx + 1}")

   x = Reduction_A(x, base_filters=base_filters, name="reduction_a")

   for idx in range(num_b_blocks):
      x = Inception_B(x, base_filters=base_filters, name=f"inception_b{idx + 1}")

   x = Reduction_B(x, base_filters=base_filters, name="reduction_b")

   for idx in range(num_c_blocks):
      x = Inception_C(x, base_filters=base_filters, name=f"inception_c{idx + 1}")

   x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

   inputs = [image_input]
   if use_gender:
      gender_input = layers.Input(shape=(), dtype=tf.int32, name="gender")
      gender_embed = layers.Embedding(input_dim=2, output_dim=8, name="gender_embed")(gender_input)
      gender_embed = layers.Flatten(name="gender_embed_flat")(gender_embed)
      x = layers.Concatenate(name="features_with_gender")([x, gender_embed])
      inputs.append(gender_input)

   if head_dropout > 0:
      x = layers.Dropout(rate=head_dropout, name="head_dropout")(x)

   x = layers.Dense(units=head_dense_units, activation="relu", name="head_dense")(x)
   age_output = layers.Dense(units=1, activation="linear", dtype=tf.float32, name="age_months")(x)

   name = "Inception_CNN_with_gender" if use_gender else "Inception_CNN"
   return Model(inputs=inputs, outputs=age_output, name=name)

# --------------------------------------
# Helper blocks for Inception modules
# --------------------------------------

def Conv_BN_ReLU(
   x: tf.Tensor,
   filters: int,
   kernel_size,
   strides: int = 1,
   padding: str = "same",
   name: Optional[str] = None,
) -> tf.Tensor:
   """
   Applies a sequence of Conv2D, BatchNormalization, and ReLU activation.
   
   Args:
      x (tf.Tensor): Input tensor.
      filters (int): Number of filters for the Conv2D layer.
      kernel_size: Kernel size for the Conv2D layer.
      strides (int): Strides for the Conv2D layer. Default is 1.
      padding (str): Padding mode. Default is "same".
      name (Optional[str]): Name prefix for the layers.
   
   Returns:
      tf.Tensor: Output tensor after Conv2D -> BN -> ReLU.
   """
   """Conv2D -> BatchNorm -> ReLU helper."""
   x = layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      use_bias=False,
      name=f"{name}_conv" if name else None,
   )(x)
   x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
   return layers.ReLU(name=f"{name}_relu" if name else None)(x)

# Stem block
def Stem(x: tf.Tensor, base_filters: int) -> tf.Tensor:
   """
   Builds the Inception-v4 stem block with factorized downsamples.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
   
   Returns:
      tf.Tensor: Output tensor after stem processing.
   """
   """Inception-v4 stem with factorized downsamples."""
   x = Conv_BN_ReLU(x, filters=base_filters, kernel_size=3, strides=2, padding="same", name="stem_conv1")
   x = Conv_BN_ReLU(x, filters=base_filters, kernel_size=3, padding="same", name="stem_conv2")
   x = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=3, padding="same", name="stem_conv3")

   b1 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool1")(x)
   b2 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=3, strides=2, padding="same", name="stem_down1")
   x = layers.Concatenate(name="stem_concat1")([b1, b2])

   x = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name="stem_conv4")
   x = Conv_BN_ReLU(x, filters=base_filters * 3, kernel_size=3, padding="same", name="stem_conv5")

   b3 = Conv_BN_ReLU(x, filters=base_filters * 3, kernel_size=3, strides=2, padding="same", name="stem_down2")
   b4 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool2")(x)
   return layers.Concatenate(name="stem_concat2")([b3, b4])

# Inception-A block
def Inception_A(x: tf.Tensor, base_filters: int, name: str) -> tf.Tensor:
   """
   Builds the 35x35 grid Inception-A block from Inception-v4.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Inception-A processing.
   """
   """35x35 grid Inception-A block (v4)."""
   b1 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 2, kernel_size=3, name=f"{name}_b2")

   b3 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b3_reduce")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 3, kernel_size=3, name=f"{name}_b3a")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 3, kernel_size=3, name=f"{name}_b3b")

   b4 = layers.AveragePooling2D(pool_size=3, strides=1, padding="same", name=f"{name}_b4_pool")(x)
   b4 = Conv_BN_ReLU(b4, filters=base_filters, kernel_size=1, name=f"{name}_b4_proj")

   return layers.Concatenate(name=f"{name}_concat")([b1, b2, b3, b4])

# Reduction-A block
def Reduction_A(x: tf.Tensor, base_filters: int, name: str) -> tf.Tensor:
   """
   Builds the Reduction-A block from Inception-v4.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Reduction-A processing.
   """
   """Reduction-A block from Inception-v4."""
   b1 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=3, strides=2, padding="same", name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 3, kernel_size=3, padding="same", name=f"{name}_b2a")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 4, kernel_size=3, strides=2, padding="same", name=f"{name}_b2b")

   b3 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name=f"{name}_b3_pool")(x)

   return layers.Concatenate(name=f"{name}_concat")([b1, b2, b3])

# Inception-B block
def Inception_B(x: tf.Tensor, base_filters: int, name: str) -> tf.Tensor:
   """
   Builds the 17x17 grid Inception-B block with factorized convolutions from Inception-v4.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Inception-B processing.
   """
   """17x17 grid Inception-B block with factorized convolutions (v4)."""
   b1 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=1, name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 4, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 5, kernel_size=(1, 7), name=f"{name}_b2a")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 5, kernel_size=(7, 1), name=f"{name}_b2b")

   b3 = Conv_BN_ReLU(x, filters=base_filters * 4, kernel_size=1, name=f"{name}_b3_reduce")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(7, 1), name=f"{name}_b3a")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(1, 7), name=f"{name}_b3b")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(7, 1), name=f"{name}_b3c")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(1, 7), name=f"{name}_b3d")

   b4 = layers.AveragePooling2D(pool_size=3, strides=1, padding="same", name=f"{name}_b4_pool")(x)
   b4 = Conv_BN_ReLU(b4, filters=base_filters * 2, kernel_size=1, name=f"{name}_b4_proj")

   return layers.Concatenate(name=f"{name}_concat")([b1, b2, b3, b4])

# Reduction-B block
def Reduction_B(x: tf.Tensor, base_filters: int, name: str) -> tf.Tensor:
   """
   Builds the Reduction-B block from Inception-v4 (17x17 -> 8x8).
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Reduction-B processing.
   """
   """Reduction-B block from Inception-v4 (17x17 -> 8x8)."""
   b1 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=1, name=f"{name}_b1_reduce")
   b1 = Conv_BN_ReLU(b1, filters=base_filters * 8, kernel_size=3, strides=2, padding="same", name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 6, kernel_size=(1, 7), name=f"{name}_b2a")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 7, kernel_size=(7, 1), name=f"{name}_b2b")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 8, kernel_size=3, strides=2, padding="same", name=f"{name}_b2c")

   b3 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name=f"{name}_b3_pool")(x)

   return layers.Concatenate(name=f"{name}_concat")([b1, b2, b3])

# Inception-C block
def Inception_C(x: tf.Tensor, base_filters: int, name: str) -> tf.Tensor:
   """
   Builds the 8x8 grid Inception-C block from Inception-v4 with split convolutions.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Inception-C processing.
   """
   """8x8 grid Inception-C block (v4) with split convolutions."""
   b1 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=1, name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 5, kernel_size=1, name=f"{name}_b2_reduce")
   b2a = Conv_BN_ReLU(b2, filters=base_filters * 6, kernel_size=(1, 3), name=f"{name}_b2a")
   b2b = Conv_BN_ReLU(b2, filters=base_filters * 6, kernel_size=(3, 1), name=f"{name}_b2b")
   b2 = layers.Concatenate(name=f"{name}_b2_concat")([b2a, b2b])

   b3 = Conv_BN_ReLU(x, filters=base_filters * 5, kernel_size=1, name=f"{name}_b3_reduce")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 7, kernel_size=(3, 1), name=f"{name}_b3a")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 7, kernel_size=(1, 3), name=f"{name}_b3b")
   b3a = Conv_BN_ReLU(b3, filters=base_filters * 7, kernel_size=(1, 3), name=f"{name}_b3c")
   b3b = Conv_BN_ReLU(b3, filters=base_filters * 7, kernel_size=(3, 1), name=f"{name}_b3d")
   b3 = layers.Concatenate(name=f"{name}_b3_concat")([b3a, b3b])

   b4 = layers.AveragePooling2D(pool_size=3, strides=1, padding="same", name=f"{name}_b4_pool")(x)
   b4 = Conv_BN_ReLU(b4, filters=base_filters * 3, kernel_size=1, name=f"{name}_b4_proj")

   return layers.Concatenate(name=f"{name}_concat")([b1, b2, b3, b4])