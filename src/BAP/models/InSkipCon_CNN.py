"""
This module provides functionality to build an Inception-ResNet Convolutional Neural Network (CNN) model
with skip connections for bone age prediction. The model is based on Inception-ResNet-v2 architecture
and can optionally include gender information as an additional input to predict age in months.
"""

from typing import Optional, Tuple

import tensorflow as tf
from keras import Model, layers


def build_InSkipConCNN(
   input_shape: Tuple[int, int, int] = (512, 512, 1),
   base_filters: int = 32,
   num_a_blocks: int = 5,
   num_b_blocks: int = 10,
   num_c_blocks: int = 5,
   scale_a: float = 0.17,
   scale_b: float = 0.1,
   scale_c: float = 0.2,
   head_dense_units: int = 128,
   head_dropout: float = 0.3,
   use_gender: bool = False,
) -> Model:
   """
   Builds an Inception-ResNet CNN model with skip connections for bone age prediction.
   
   Args:
      input_shape (Tuple[int, int, int]): Shape of the input image (height, width, channels). Default is (512, 512, 1).
      base_filters (int): Base number of filters used in the model. Default is 32.
      num_a_blocks (int): Number of InceptionResNet-A blocks. Default is 5.
      num_b_blocks (int): Number of InceptionResNet-B blocks. Default is 10.
      num_c_blocks (int): Number of InceptionResNet-C blocks. Default is 5.
      scale_a (float): Scaling factor for residual in A blocks. Default is 0.17.
      scale_b (float): Scaling factor for residual in B blocks. Default is 0.1.
      scale_c (float): Scaling factor for residual in C blocks. Default is 0.2.
      head_dense_units (int): Number of units in the head dense layer. Default is 128.
      head_dropout (float): Dropout rate for the head. Default is 0.3.
      use_gender (bool): Whether to include gender as an additional input. Default is False.
   
   Returns:
      Model: A Keras Model instance for bone age prediction.
   """
   image_input = layers.Input(shape=input_shape, dtype=tf.float32, name="image")

   x = Stem(image_input, base_filters=base_filters)

   for idx in range(num_a_blocks):
      x = InceptionResNet_A(
         x,
         base_filters=base_filters,
         scale=scale_a,
         name=f"inception_resnet_a{idx + 1}",
      )

   x = Reduction_A(x, base_filters=base_filters, name="reduction_a")

   for idx in range(num_b_blocks):
      x = InceptionResNet_B(
         x,
         base_filters=base_filters,
         scale=scale_b,
         name=f"inception_resnet_b{idx + 1}",
      )

   x = Reduction_B(x, base_filters=base_filters, name="reduction_b")

   for idx in range(num_c_blocks):
      x = InceptionResNet_C(
         x,
         base_filters=base_filters,
         scale=scale_c,
         name=f"inception_resnet_c{idx + 1}",
      )

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

   name = "InSkipCon_CNN_with_gender" if use_gender else "InSkipCon_CNN"
   return Model(inputs=inputs, outputs=age_output, name=name)


# --------------------------------------
# Helper layers / blocks
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


def _residual_add(
   x: tf.Tensor,
   residual: tf.Tensor,
   scale: float,
   name: str,
) -> tf.Tensor:
   """
   Scales the residual branch and adds it to the shortcut connection.
   
   Args:
      x (tf.Tensor): Shortcut tensor.
      residual (tf.Tensor): Residual tensor to be added.
      scale (float): Scaling factor for the residual.
      name (str): Name prefix for the operations.
   
   Returns:
      tf.Tensor: Output tensor after adding residual and applying ReLU.
   """
   """Scale the residual branch and add it to the shortcut."""
   if scale != 1.0:
      residual = layers.Lambda(lambda t: t * scale, name=f"{name}_scale")(residual)
   x = layers.Add(name=f"{name}_add")([x, residual])
   return layers.ReLU(name=f"{name}_out")(x)


def Stem(x: tf.Tensor, base_filters: int) -> tf.Tensor:
   """
   Builds the Inception-ResNet-v2 stem block with factorized downsamples.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
   
   Returns:
      tf.Tensor: Output tensor after stem processing.
   """
   """Inception-ResNet-v2 stem with factorized downsamples."""
   x = Conv_BN_ReLU(x, filters=base_filters, kernel_size=3, strides=2, padding="same", name="stem_conv1")
   x = Conv_BN_ReLU(x, filters=base_filters, kernel_size=3, padding="same", name="stem_conv2")
   x = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=3, padding="same", name="stem_conv3")

   b1 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool1")(x)
   b2 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=3, strides=2, padding="same", name="stem_down1")
   x = layers.Concatenate(name="stem_concat1")([b1, b2])

   b3 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name="stem_conv4")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 3, kernel_size=3, strides=2, name="stem_conv5")

   b4 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name="stem_path2_conv1")
   b4 = Conv_BN_ReLU(b4, filters=base_filters * 3, kernel_size=3, name="stem_path2_conv2")
   b4 = Conv_BN_ReLU(b4, filters=base_filters * 3, kernel_size=3, strides=2, padding="same", name="stem_path2_down")

   b5 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool2")(x)
   x = layers.Concatenate(name="stem_concat2")([b3, b4, b5])

   return x


def InceptionResNet_A(
   x: tf.Tensor,
   base_filters: int,
   scale: float,
   name: str,
) -> tf.Tensor:
   """
   Builds the 35x35 grid Inception-ResNet-A block.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      scale (float): Scaling factor for the residual.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Inception-ResNet-A processing.
   """
   """35x35 grid Inception-ResNet-A block."""
   shortcut = x

   b1 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 2, kernel_size=3, name=f"{name}_b2")

   b3 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b3_reduce")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 3, kernel_size=3, name=f"{name}_b3a")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 4, kernel_size=3, name=f"{name}_b3b")

   mixed = layers.Concatenate(name=f"{name}_concat")([b1, b2, b3])
   up = layers.Conv2D(
      filters=int(shortcut.shape[-1]),
      kernel_size=1,
      padding="same",
      use_bias=False,
      name=f"{name}_proj",
   )(mixed)
   up = layers.BatchNormalization(name=f"{name}_proj_bn")(up)
   return _residual_add(shortcut, up, scale=scale, name=name)


def InceptionResNet_B(
   x: tf.Tensor,
   base_filters: int,
   scale: float,
   name: str,
) -> tf.Tensor:
   """
   Builds the 17x17 grid Inception-ResNet-B block with factorized convolutions.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      scale (float): Scaling factor for the residual.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Inception-ResNet-B processing.
   """
   """17x17 grid Inception-ResNet-B block with factorized convolutions."""
   shortcut = x

   b1 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=1, name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 4, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 5, kernel_size=(1, 7), name=f"{name}_b2a")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 5, kernel_size=(7, 1), name=f"{name}_b2b")

   b3 = Conv_BN_ReLU(x, filters=base_filters * 4, kernel_size=1, name=f"{name}_b3_reduce")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(7, 1), name=f"{name}_b3a")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(1, 7), name=f"{name}_b3b")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(7, 1), name=f"{name}_b3c")
   b3 = Conv_BN_ReLU(b3, filters=base_filters * 6, kernel_size=(1, 7), name=f"{name}_b3d")

   mixed = layers.Concatenate(name=f"{name}_concat")([b1, b2, b3])
   up = layers.Conv2D(
      filters=int(shortcut.shape[-1]),
      kernel_size=1,
      padding="same",
      use_bias=False,
      name=f"{name}_proj",
   )(mixed)
   up = layers.BatchNormalization(name=f"{name}_proj_bn")(up)
   return _residual_add(shortcut, up, scale=scale, name=name)


def InceptionResNet_C(
   x: tf.Tensor,
   base_filters: int,
   scale: float,
   name: str,
) -> tf.Tensor:
   """
   Builds the 8x8 grid Inception-ResNet-C block.
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      scale (float): Scaling factor for the residual.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Inception-ResNet-C processing.
   """
   """8x8 grid Inception-ResNet-C block."""
   shortcut = x

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

   mixed = layers.Concatenate(name=f"{name}_concat")([b1, b2, b3])
   up = layers.Conv2D(
      filters=int(shortcut.shape[-1]),
      kernel_size=1,
      padding="same",
      use_bias=False,
      name=f"{name}_proj",
   )(mixed)
   up = layers.BatchNormalization(name=f"{name}_proj_bn")(up)
   return _residual_add(shortcut, up, scale=scale, name=name)


def Reduction_A(x: tf.Tensor, base_filters: int, name: str) -> tf.Tensor:
   """
   Builds the Reduction-A block (35x35 -> 17x17).
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Reduction-A processing.
   """
   """Reduction-A block (35x35 -> 17x17)."""
   b1 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=3, strides=2, padding="same", name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 2, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 3, kernel_size=3, padding="same", name=f"{name}_b2a")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 4, kernel_size=3, strides=2, padding="same", name=f"{name}_b2b")

   b3 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name=f"{name}_b3_pool")(x)

   return layers.Concatenate(name=f"{name}_concat")([b1, b2, b3])


def Reduction_B(x: tf.Tensor, base_filters: int, name: str) -> tf.Tensor:
   """
   Builds the Reduction-B block (17x17 -> 8x8).
   
   Args:
      x (tf.Tensor): Input tensor.
      base_filters (int): Base number of filters.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after Reduction-B processing.
   """
   """Reduction-B block (17x17 -> 8x8)."""
   b1 = Conv_BN_ReLU(x, filters=base_filters * 6, kernel_size=1, name=f"{name}_b1_reduce")
   b1 = Conv_BN_ReLU(b1, filters=base_filters * 8, kernel_size=3, strides=2, padding="same", name=f"{name}_b1")

   b2 = Conv_BN_ReLU(x, filters=base_filters * 5, kernel_size=1, name=f"{name}_b2_reduce")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 6, kernel_size=(1, 7), name=f"{name}_b2a")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 6, kernel_size=(7, 1), name=f"{name}_b2b")
   b2 = Conv_BN_ReLU(b2, filters=base_filters * 7, kernel_size=3, strides=2, padding="same", name=f"{name}_b2c")

   b3 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name=f"{name}_b3_pool")(x)

   return layers.Concatenate(name=f"{name}_concat")([b1, b2, b3])
