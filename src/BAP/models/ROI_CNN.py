"""Region-of-interest CNN components for wrist bone-age regression.

The module exposes:
1. `build_ROI_CNN`, which wires together carpal and metacarpal ROI branches
   (and an optional gender embedding) to output a scalar age prediction.
2. `ROI_CNN_head`, a reusable convolutional encoder used by each branch for
   retaining spatial context.
"""

from typing import Sequence, Tuple
import tensorflow as tf
from keras import layers, Model
#from BAP.utils.logger import get_logger

#logger = get_logger(__name__)

def build_ROI_CNN(
   roi_shape: Tuple[int, int, int] = (224, 224, 1),
   channels: Sequence[int] = [32, 64],
   dense_units: int = 32,
   dropout_rate: float = 0.2,
   use_gender: bool = False,
) -> Model:
   """Assemble the dual-branch ROI CNN and return a Keras model.

   The carpal and metacarpal crops pass through identical ROI heads, their
   embeddings are concatenated, and an optional gender embedding can be
   appended before the final dense layers emit the age estimate. The function
   keeps the graph compact but configurable, allowing callers to tweak
   channel widths, dense size, or dropout.

   Args:
      roi_shape: Spatial shape for each ROI input tensor `(H, W, C)`.
      channels: Sequence of filter counts per convolutional block in a head.
      dense_units: Number of units in each ROI head's dense projection.
      dropout_rate: Drop probability applied to concatenated features; set to
         `0` to disable.
      use_gender: If True, expect an additional gender input and inject its 
      learned embedding into the pooled features.

   Returns:
      Keras `Model` that maps ROI inputs (and optional gender) to age in months.
   """
   
   input_carp = layers.Input(shape=roi_shape, name="carpal") # [B,H,W,1]
   input_metp = layers.Input(shape=roi_shape, name="metaph") # [B,H,W,1]

   feat_carp = ROI_CNN_head(input_carp, channels, dense_units, "carpal") # [B,dense_units]
   feat_metp = ROI_CNN_head(input_metp, channels, dense_units, "metaph") # [B,dense_units]

   features = layers.Concatenate(name="ROI_heads_concat")([feat_carp, feat_metp]) # [B,dense_units*2]
   if use_gender:
      inp_gender = layers.Input(shape=(), dtype=tf.int32, name="gender") # [B,]
      gender_emb = layers.Embedding(input_dim=2, output_dim=8, name="gender_embed")(inp_gender) # [B,8]
      gender_emb = layers.Flatten(name="gender_embed_flat")(gender_emb) # [B,8]
      features = layers.Concatenate(name="ROI_heads_concat_with_gender")([features, gender_emb]) # [B,dense_units*2+8]
      inputs = [input_carp, input_metp, inp_gender] 
      name = "ROI_CNN_with_gender"
      #logger.info("Building ROI-CNN model with gender input.")
   else:
      inputs = [input_carp, input_metp]
      name = "ROI_CNN"
      #logger.info("Building ROI-CNN model w/o gender input.")

   if dropout_rate > 0:
      x = layers.Dropout(rate=dropout_rate, name="dropout")(features) # [B,dense_units*2(+8)]
   x = layers.Dense(
      units=max(64, dense_units * 2), 
      activation="relu",
      name="dense"
   )(x) # [B, max(64, dense_units*2)]
   output_age = layers.Dense(units=1, activation="linear", name="age_months", dtype=tf.float32)(x) # [B,1]

   model = Model(inputs=inputs, outputs=output_age, name=name)
   return model



def ROI_CNN_head (
   x: tf.Tensor, 
   channels: Sequence[int] = [32, 64],
   dense_units: int = 32,
   name: str = "carpal"
) -> tf.Tensor:
   """Encode an ROI tensor into a compact feature vector.

   Each entry in `channels` creates a block with two Conv-BN-ReLU layers
   followed by 2x2 max pooling, progressively reducing the spatial resolution
   and increasing the receptive field. After the blocks, the tensor goes through
   global average pooling and a dense projection that yields the final ROI
   embedding suitable for concatenation within `build_ROI_CNN`.

   Args:
      x: Input ROI tensor shaped `[batch, height, width, channels]`.
      channels: Filter counts assigned to consecutive convolutional blocks.
      dense_units: Size of the dense layer applied post pooling.
      name: Prefix added to layer names for traceability when multiple heads
         coexist in the same model.

   Returns:
      Tensor of shape `[batch, dense_units]` representing the ROI features.
   """


   for i, ch in enumerate(channels):
      x = layers.Conv2D(
         filters=ch, 
         kernel_size=3, 
         padding='same', 
         use_bias=False,
         name=f"{name}_roi_block{i+1}_conv1"
      )(x) # [B,H,W,ch]
      x = layers.BatchNormalization(name=f"{name}_roi_block{i+1}_bn1")(x) # [B,H,W,ch]
      x = layers.ReLU(name=f"{name}_roi_block{i+1}_relu1")(x) # [B,H,W,ch]
      
      x = layers.Conv2D(
         filters=ch, 
         kernel_size=3, 
         padding='same', 
         use_bias=False,
         name=f"{name}_roi_block{i+1}_conv2"
      )(x) # [B,H,W,ch]
      x = layers.BatchNormalization(name=f"{name}_roi_block{i+1}_bn2")(x) # [B,H,W,ch]
      x = layers.ReLU(name=f"{name}_roi_block{i+1}_relu2")(x) # [B,H,W,ch]      
      
      x = layers.MaxPool2D(pool_size=2, padding='same', name=f"{name}_roi_block{i+1}_pool")(x) # [B,H/2,W/2,ch]

   x = layers.GlobalAveragePooling2D(name=f"{name}_roi_global_avg_pool")(x) # [B,ch]
   x = layers.Dense(dense_units, activation='relu', name=f"{name}_roi_dense")(x) # [B,dense_units]
   return x
