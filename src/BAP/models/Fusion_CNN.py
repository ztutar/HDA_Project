#TODO: Add docstring explanation for this module. Write in details and explain each function and class.

from typing import Sequence, Tuple

import tensorflow as tf
from keras import layers, Model

from BAP.models.ROI_CNN import ROI_CNN_head
from BAP.utils.logger import get_logger

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
   image_input = layers.Input(shape=global_input_shape, dtype=tf.float32, name="image") # [B,H,W,1]
   global_features = _global_branch(image_input, global_channels, global_dense_units) # [B,global_dense_units]

   carpal_input = layers.Input(shape=roi_shape, dtype=tf.float32, name="carpal") # [B,H,W,1]
   metaph_input = layers.Input(shape=roi_shape, dtype=tf.float32, name="metaph") # [B,H,W,1]
   carpal_features = ROI_CNN_head(carpal_input, roi_channels, roi_dense_units, "carpal") # [B,roi_dense_units]
   metaph_features = ROI_CNN_head(metaph_input, roi_channels, roi_dense_units, "metaph") # [B,roi_dense_units]

   fused = layers.Concatenate(name="fusion_concat")([global_features, carpal_features, metaph_features]) # [B, global_dense_units + roi_dense_units*2]
   inputs = {
      "image": image_input,
      "carpal": carpal_input,
      "metaph": metaph_input,
   }

   if use_gender:
      gender_input = layers.Input(shape=(), dtype=tf.int32, name="gender") # [B,]
      gender_embedding = layers.Embedding(input_dim=2, output_dim=8, name="gender_embed")(gender_input) # [B,8]
      gender_embedding = layers.Flatten(name="gender_embed_flat")(gender_embedding) # [B,8]
      fused = layers.Concatenate(name="fusion_concat_with_gender")([fused, gender_embedding]) # [B, global_dense_units + roi_dense_units*2 + 8]
      inputs["gender"] = gender_input 
      name = "Fusion_CNN_with_gender"
      logger.info("Building Fusion CNN model with gender input.")
   else:
      name = "Fusion_CNN"
      logger.info("Building Fusion CNN model w/o gender input.")

   x = fused

   for idx, units in enumerate(fusion_dense_units):
      if dropout_rate > 0:
         x = layers.Dropout(rate=dropout_rate, name=f"fusion_dropout_{idx + 1}")(x) # [B, units]
      x = layers.Dense(units=units, activation="relu", name=f"fusion_dense_{idx + 1}")(x) # [B, units]

   output = layers.Dense(units=1, activation="linear", name="age_months", dtype=tf.float32)(x) # [B,1]

   model = Model(inputs=inputs, outputs=output, name=name)
   logger.info("Building Fusion CNN model%s.", " with gender input" if use_gender else "")
   return model


def _global_branch(
   x: tf.Tensor,
   channels: Sequence[int],
   dense_units: int,
) -> tf.Tensor:
   for idx, ch in enumerate(channels):
      x = layers.Conv2D(
         filters=ch, 
         kernel_size=3, 
         strides=1, 
         padding="same", 
         use_bias=False, 
         name=f"global_block{idx + 1}_conv1"
      )(x) # [B,H,W,ch]
      x = layers.BatchNormalization(name=f"global_block{idx + 1}_bn1")(x) # [B,H,W,ch]
      x = layers.ReLU(name=f"global_block{idx + 1}_relu1")(x) # [B,H,W,ch]

      x = layers.Conv2D(
         filters=ch, 
         kernel_size=3, 
         padding="same", 
         use_bias=False, 
         name=f"global_block{idx + 1}_conv2"
      )(x) # [B,H,W,ch]
      x = layers.BatchNormalization(name=f"global_block{idx + 1}_bn2")(x) # [B,H,W,ch]
      x = layers.ReLU(name=f"global_block{idx + 1}_relu2")(x) # [B,H,W,ch]

      x = layers.MaxPooling2D(pool_size=2, padding="same", name=f"global_block{idx + 1}_pool")(x) # [B,H/2,W/2,ch]

   x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x) # [B,ch]
   x = layers.Dense(dense_units, activation="relu", name="global_dense")(x) # [B,dense_units]
   return x
