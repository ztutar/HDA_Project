"""
Fusion CNN module that combines a global full-hand branch with ROI-specific CNN
heads to predict skeletal age. The module exposes two public helpers:

- ``build_FusionCNN`` builds a multi-input Keras model that ingests a full-hand
   radiograph along with cropped carpal and metacarpal regions (and optionally a
   gender indicator). The function wires the shared ROI heads, fuses feature
   vectors, applies configurable dense/dropout layers, and produces the final
   regression output in months. All hyperparameters (channel counts, dense units,
   dropout rates, fusion stack sizes, and the gender embedding) are surfaced so
   callers can tailor the network architecture without touching the internals.

- ``_global_branch`` defines the convolutional feature extractor used for the
   full-hand image. It stacks configurable Conv-BN-ReLU blocks with max pooling
   for progressive downsampling, then applies global average pooling and a dense
   layer to deliver a compact descriptor that feeds into the fusion stage. While
   intended as an internal helper, documenting it here clarifies the exact flow
   and makes it easier to audit or extend the backbone.
"""

from typing import Sequence, Tuple

import tensorflow as tf
from keras import layers, Model

from BAP.models.ROI_CNN import ROI_CNN_head
#from BAP.utils.logger import get_logger

#logger = get_logger(__name__)

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
   Build a multi-input Keras model that fuses a full-hand CNN branch with ROI
   heads (carpal and metacarpal) and optional gender embeddings to regress bone
   age in months. The helper exposes the key architectural knobs so experiments
   can scale channel widths, dense units, dropout rate, and gender usage without
   touching the model internals.
   
   Args:
      global_input_shape: Tensor shape of the whole-hand image fed into the
         global branch.
      roi_shape: Input shape shared by the carpal and metacarpal ROI crops.
      global_channels: Per-block filter counts for the global feature extractor.
      roi_channels: Channel configuration passed to each ROI head.
      global_dense_units: Size of the dense projection after the global branch.
      roi_dense_units: Size of the dense layer inside each ROI head.
      fusion_dense_units: Sequence of dense layer widths applied after feature
         concatenation.
      dropout_rate: Dropout probability applied before each fusion dense layer.
      use_gender: If True, expect an additional gender input and inject its 
      learned embedding into the pooled features.
   
   Returns:
      Compiled Keras `Model` ready to train for bone-age regression.
   """
   image_input = layers.Input(shape=global_input_shape, dtype=tf.float32, name="image") # [B,H,W,1]
   global_features = _global_branch(image_input, global_channels, global_dense_units) # [B,global_dense_units]

   carpal_input = layers.Input(shape=roi_shape, dtype=tf.float32, name="carpal") # [B,H,W,1]
   metaph_input = layers.Input(shape=roi_shape, dtype=tf.float32, name="metaph") # [B,H,W,1]
   carpal_features = ROI_CNN_head(carpal_input, roi_channels, roi_dense_units, "carpal") # [B,roi_dense_units]
   metaph_features = ROI_CNN_head(metaph_input, roi_channels, roi_dense_units, "metaph") # [B,roi_dense_units]

   fused = layers.Concatenate(name="fusion_concat")([global_features, carpal_features, metaph_features]) # [B, global_dense_units + roi_dense_units*2]

   if use_gender:
      gender_input = layers.Input(shape=(), dtype=tf.int32, name="gender") # [B,]
      gender_embedding = layers.Embedding(input_dim=2, output_dim=8, name="gender_embed")(gender_input) # [B,8]
      gender_embedding = layers.Flatten(name="gender_embed_flat")(gender_embedding) # [B,8]
      fused = layers.Concatenate(name="fusion_concat_with_gender")([fused, gender_embedding]) # [B, global_dense_units + roi_dense_units*2 + 8]
      inputs = [image_input, carpal_input, metaph_input, gender_input]
      name = "Fusion_CNN_with_gender"
      #logger.info("Building Fusion CNN model with gender input.")
   else:
      inputs = [image_input, carpal_input, metaph_input]
      name = "Fusion_CNN"
      #logger.info("Building Fusion CNN model w/o gender input.")

   x = fused

   for idx, units in enumerate(fusion_dense_units):
      if dropout_rate > 0:
         x = layers.Dropout(rate=dropout_rate, name=f"fusion_dropout_{idx + 1}")(x) # [B, units]
      x = layers.Dense(units=units, activation="relu", name=f"fusion_dense_{idx + 1}")(x) # [B, units]

   output = layers.Dense(units=1, activation="linear", name="age_months", dtype=tf.float32)(x) # [B,1]

   model = Model(inputs=inputs, outputs=output, name=name)
   #logger.info("Building Fusion CNN model%s.", " with gender input" if use_gender else "")
   return model


def _global_branch(

   x: tf.Tensor,
   channels: Sequence[int],
   dense_units: int,
) -> tf.Tensor:
   """
   Build the convolutional backbone that processes the full-hand radiograph.

   Args:
      x: Keras tensor for the global image input.
      channels: Filter counts for each Conv-BN-ReLU block; a max-pooling layer
         follows every block to halve the spatial resolution while increasing
         representational capacity.
      dense_units: Width of the dense projection applied after global average
         pooling to produce the feature vector that feeds the fusion head.

   Returns:
      Tensor representing the per-image descriptor emitted by the global branch.
   """
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
