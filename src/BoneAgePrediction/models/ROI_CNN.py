#TODO: add docstring explanation for this module. write in details and explain each function and class


from typing import Sequence, Tuple
import tensorflow as tf
from keras import layers, Model
from BoneAgePrediction.utils.logger import get_logger

logger = get_logger(__name__)

def build_ROI_CNN(
   roi_shape: Tuple[int, int, int] = (224, 224, 1),
   channels: Sequence[int] = [32, 64],
   dense_units: int = 32,
   use_gender: bool = False,
) -> Model:
   """
   Build the ROI-only model with two inputs (carpal, metacarpal/phalange).
   Features from both heads are concatenated and regressed to age.

   Args:
      roi_shape (Tuple[int, int, int], optional): Per-ROI crop shape. Defaults to (224, 224, 1).
      channels (Sequence[int], optional): Conv channels for each head stage. Defaults to [32, 64].
      dense_units (int, optional): Dense units for each head embedding. Defaults to 32.
      use_gender (bool, optional): If True, include small gender embedding. Defaults to False.

   Returns:
      Model: Compiled model mapping ROIs -> age (months).
                  Inputs: {"carpal": [B,H,W,1], "metaph": [B,H,W,1], "gender": [B,]}
                  Output: [B,1] age (months)  
   """
   
   input_carp = layers.Input(shape=roi_shape, name="carpal") # [B,H,W,1]
   input_metp = layers.Input(shape=roi_shape, name="metaph") # [B,H,W,1]
   inputs = {"carpal": input_carp, "metaph": input_metp} 

   feat_carp = ROI_CNN_head(input_carp, channels, dense_units) # [B,dense_units]
   feat_metp = ROI_CNN_head(input_metp, channels, dense_units) # [B,dense_units]

   features = layers.Concatenate()([feat_carp, feat_metp]) # [B,dense_units*2]
   if use_gender:
      inp_gender = layers.Input(shape=(), dtype=tf.int32, name="gender") # [B,]
      gender_emb = layers.Embedding(input_dim=2, output_dim=8)(inp_gender) # [B,8]
      gender_emb = layers.Flatten(name="gender_emb")(gender_emb) # [B,8]
      features = layers.Concatenate(name="features_with_gender")([features, gender_emb]) # [B,dense_units*2+8]
      inputs["gender"] = inp_gender 
      name = "ROI_CNN_with_gender"
      logger.info("Building ROI-CNN model with gender input.")
   else:
      name = "ROI_CNN"
      logger.info("Building ROI-CNN model w/o gender input.")

   x = layers.Dropout(rate=0.2)(features) # [B,dense_units*2(+8)]
   x = layers.Dense(units=max(64, dense_units * 2), activation="relu")(x) # [B, max(64, dense_units*2)]
   output_age = layers.Dense(units=1, activation="linear", name="age_months", dtype=tf.float32)(x) # [B,1]

   model = Model(inputs=inputs, outputs=output_age, name=name)
   return model



def ROI_CNN_head (
   x: tf.Tensor, 
   channels: Sequence[int] = [32, 64],
   dense_units: int = 32
) -> tf.Tensor:
   """
   A CNN head for a ROI crop.

   Args:
      x (tf.Tensor): ROI image [B, H, W, 1] float32.
      channels (Sequence[int]): e.g. [32, 64]
      dense_units (int): Units in the per-ROI embedding.

   Returns:
      tf.Tensor: ROI feature embedding [B, dense_units].
   """

   for i, ch in enumerate(channels):
      x = layers.Conv2D(ch, 3, padding='same', use_bias=False)(x) # [B,H,W,ch]
      x = layers.BatchNormalization()(x) # [B,H,W,ch]
      x = layers.ReLU()(x) # [B,H,W,ch]
      
      x = layers.Conv2D(ch, 3, padding='same', use_bias=False)(x) # [B,H,W,ch]
      x = layers.BatchNormalization()(x) # [B,H,W,ch]
      x = layers.ReLU()(x) # [B,H,W,ch]      
      
      x = layers.MaxPool2D(pool_size=2, padding='same')(x) # [B,H/2,W/2,ch]

   x = layers.GlobalAveragePooling2D()(x) # [B,ch]
   x = layers.Dense(dense_units, activation='relu')(x) # [B,dense_units]
   return x
