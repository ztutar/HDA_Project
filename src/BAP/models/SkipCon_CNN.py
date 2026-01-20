"""
This module provides functionality to build a Skip Connection Convolutional Neural Network (CNN) model
for bone age prediction. The model uses residual blocks with skip connections and can optionally
include gender information as an additional input to predict age in months.
"""

from typing import Sequence, Tuple

import tensorflow as tf
from keras import Model, layers


def build_SkipConCNN(
   input_shape: Tuple[int, int, int] = (512, 512, 1),
   base_filters: int = 32,
   block_filters: Sequence[int] = (32, 64, 128, 256),
   blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
   dense_units: int = 128,
   dropout_rate: float = 0.2,
   use_gender: bool = False,
) -> Model:
   """
   Builds a Skip Connection CNN model for bone age prediction.
   
   Args:
      input_shape (Tuple[int, int, int]): Shape of the input image (height, width, channels). Default is (512, 512, 1).
      base_filters (int): Number of filters in the stem convolution. Default is 32.
      block_filters (Sequence[int]): Number of filters for each stage. Default is (32, 64, 128, 256).
      blocks_per_stage (Sequence[int]): Number of blocks per stage. Default is (2, 2, 2, 2).
      dense_units (int): Number of units in the dense layer. Default is 128.
      dropout_rate (float): Dropout rate for regularization. Default is 0.2.
      use_gender (bool): Whether to include gender as an additional input. Default is False.
   
   Returns:
      Model: A Keras Model instance for bone age prediction.
   """
   image_input = layers.Input(shape=input_shape, dtype=tf.float32, name="image")

   # Stem
   x = layers.Conv2D(
      filters=base_filters,
      kernel_size=7,
      strides=2,
      padding="same",
      use_bias=False,
      name="stem_conv",
   )(image_input)
   x = layers.BatchNormalization(name="stem_bn")(x)
   x = layers.ReLU(name="stem_relu")(x)
   x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="stem_pool")(x)

   # Residual stages
   for stage_idx, (filters, num_blocks) in enumerate(zip(block_filters, blocks_per_stage)):
      for block_idx in range(num_blocks):
         stride = 2 if block_idx == 0 and stage_idx > 0 else 1
         x = SkipConBlock(
            x, filters=filters, 
            stride=stride,
            name=f"stage{stage_idx + 1}_block{block_idx + 1}",
            )

   x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

   inputs = [image_input]
   if use_gender:
      gender_input = layers.Input(shape=(), dtype=tf.int32, name="gender")
      gender_embed = layers.Embedding(input_dim=2, output_dim=8, name="gender_embed")(gender_input)
      gender_embed = layers.Flatten(name="gender_embed_flat")(gender_embed)
      x = layers.Concatenate(name="features_with_gender")([x, gender_embed])
      inputs.append(gender_input)

   if dropout_rate > 0:
      x = layers.Dropout(rate=dropout_rate, name="dropout")(x)

   x = layers.Dense(units=dense_units, activation="relu", name="dense")(x)
   age_output = layers.Dense(units=1, activation="linear", dtype=tf.float32, name="age_months")(x)

   name = "SkipCon_CNN_with_gender" if use_gender else "SkipCon_CNN"
   return Model(inputs=inputs, outputs=age_output, name=name)


def SkipConBlock(
   x: tf.Tensor,
   filters: int,
   stride: int,
   name: str,
) -> tf.Tensor:
   """
   Builds a basic residual block with projection skip connection when the shape changes.
   
   Args:
      x (tf.Tensor): Input tensor.
      filters (int): Number of filters for the convolutions.
      stride (int): Stride for the first convolution.
      name (str): Name prefix for the block.
   
   Returns:
      tf.Tensor: Output tensor after the residual block.
   """
   """Basic residual block with projection skip when shape changes."""
   shortcut = x

   x = layers.Conv2D(
      filters=filters,
      kernel_size=3,
      strides=stride,
      padding="same",
      use_bias=False,
      name=f"{name}_conv1",
   )(x)
   x = layers.BatchNormalization(name=f"{name}_bn1")(x)
   x = layers.ReLU(name=f"{name}_relu1")(x)

   x = layers.Conv2D(
      filters=filters,
      kernel_size=3,
      strides=1,
      padding="same",
      use_bias=False,
      name=f"{name}_conv2",
   )(x)
   x = layers.BatchNormalization(name=f"{name}_bn2")(x)

   if shortcut.shape[-1] != filters or stride != 1:
      shortcut = layers.Conv2D(
         filters=filters,
         kernel_size=1,
         strides=stride,
         padding="same",
         use_bias=False,
         name=f"{name}_proj",
      )(shortcut)
      shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

   x = layers.Add(name=f"{name}_add")([x, shortcut])
   return layers.ReLU(name=f"{name}_out")(x)
