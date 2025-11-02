#TODO: Add docstring explanation for this module. Write in details and explain each function and class.

from typing import Sequence, Tuple

import tensorflow as tf
from keras import layers, Model

from BoneAgePrediction.models.ROI_CNN import ROI_CNN_head
from BoneAgePrediction.utils.logger import get_logger

logger = get_logger(__name__)


def build_FusionCNN(
   global_input_shape: Tuple[int, int, int] = (512, 512, 1),
   roi_shape: Tuple[int, int, int] = (224, 224, 1),
   global_channels: Sequence[int] = (32, 64, 128),
   roi_channels: Sequence[int] = (32, 64),
   global_dense_units: int = 128,
   roi_dense_units: int = 32,
   fusion_dense_units: Sequence[int] = (256, 128),
   dropout_rate: float = 0.2,
   use_gender: bool = False,
) -> Model:
   """
   Build the Fusion CNN model that fuses full-hand and ROI features and predicts age.
   """
   image_input = layers.Input(shape=global_input_shape, dtype=tf.float32, name="image")
   global_features = _global_branch(image_input, global_channels, global_dense_units)

   carpal_input = layers.Input(shape=roi_shape, dtype=tf.float32, name="carpal")
   metaph_input = layers.Input(shape=roi_shape, dtype=tf.float32, name="metaph")
   carpal_features = ROI_CNN_head(carpal_input, channels=roi_channels, dense_units=roi_dense_units)
   metaph_features = ROI_CNN_head(metaph_input, channels=roi_channels, dense_units=roi_dense_units)

   fused = layers.Concatenate(name="fusion_concat")([global_features, carpal_features, metaph_features])
   inputs = {
      "image": image_input,
      "carpal": carpal_input,
      "metaph": metaph_input,
   }

   if use_gender:
      gender_input = layers.Input(shape=(), dtype=tf.int32, name="gender")
      gender_embedding = layers.Embedding(input_dim=2, output_dim=8, name="gender_embedding")(gender_input)
      gender_embedding = layers.Flatten(name="gender_emb")(gender_embedding)
      fused = layers.Concatenate(name="fusion_concat_with_gender")([fused, gender_embedding])
      inputs["gender"] = gender_input

   x = fused
   if dropout_rate > 0:
      x = layers.Dropout(rate=dropout_rate, name="fusion_dropout_0")(x)

   for idx, units in enumerate(fusion_dense_units):
      x = layers.Dense(units=units, activation="relu", name=f"fusion_dense_{idx + 1}")(x)
      if dropout_rate > 0:
         x = layers.Dropout(rate=dropout_rate, name=f"fusion_dropout_{idx + 1}")(x)

   output = layers.Dense(units=1, activation="linear", name="age_months", dtype=tf.float32)(x)

   model_name = "Fusion_CNN_with_gender" if use_gender else "Fusion_CNN"
   model = Model(inputs=inputs, outputs=output, name=model_name)
   logger.info("Building Fusion CNN model%s.", " with gender input" if use_gender else "")
   return model


def _global_branch(
   x: tf.Tensor,
   channels: Sequence[int],
   dense_units: int,
) -> tf.Tensor:
   for idx, ch in enumerate(channels):
      x = layers.Conv2D(filters=ch, kernel_size=3, strides=1, padding="same", use_bias=False, name=f"global_block{idx + 1}_conv1")(x)
      x = layers.BatchNormalization(name=f"global_block{idx + 1}_bn1")(x)
      x = layers.ReLU(name=f"global_block{idx + 1}_relu1")(x)

      x = layers.Conv2D(filters=ch, kernel_size=3, strides=1, padding="same", use_bias=False, name=f"global_block{idx + 1}_conv2")(x)
      x = layers.BatchNormalization(name=f"global_block{idx + 1}_bn2")(x)
      x = layers.ReLU(name=f"global_block{idx + 1}_relu2")(x)

      x = layers.MaxPooling2D(pool_size=2, padding="same", name=f"global_block{idx + 1}_pool")(x)

   x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
   x = layers.Dense(dense_units, activation="relu", name="global_dense")(x)
   return x
