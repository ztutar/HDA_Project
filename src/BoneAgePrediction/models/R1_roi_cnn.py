from typing import Sequence, Tuple
import tensorflow as tf
from keras import layers, Model

def build_ROI_CNN(
   roi_shape: Tuple[int, int, int] = (224, 224, 1),
   channels: Sequence[int] = [32, 64],
   dense_units: int = 32,
   use_gender: bool = False,
   name: str = "R1_ROI_CNN",
) -> tf.keras.Model:
   """
   Build the ROI-only model with two inputs (carpal, metacarpal/phalange).
   Features from both heads are concatenated and regressed to age.

   Args:
      roi_shape (Tuple[int, int, int], optional): Per-ROI crop shape. Defaults to (224, 224, 1).
      channels (Sequence[int], optional): Conv channels for each head stage. Defaults to [32, 64].
      dense_units (int, optional): Dense units for each head embedding. Defaults to 32.
      use_gender (bool, optional): If True, include small gender embedding. Defaults to False.
      name (str, optional): Model name. Defaults to 'R1_ROIOnly_CNN'.

   Returns:
      Model: Compiled model mapping ROIs -> age (months).
                  Inputs: {"carpal": [B,H,W,1], "metaph": [B,H,W,1], "gender": [B,]}
                  Output: [B,1] age (months)  
   """
   
   inp_carp = layers.Input(shape=roi_shape, name="carpal")
   inp_metp = layers.Input(shape=roi_shape, name="metaph")
   inputs = {"carpal": inp_carp, "metaph": inp_metp}

   feat_carp = ROI_CNN_head(inp_carp, channels, dense_units)
   feat_metp = ROI_CNN_head(inp_metp, channels, dense_units)

   features = layers.Concatenate()([feat_carp, feat_metp])
   if use_gender:
      inp_gender = layers.Input(shape=(), dtype=tf.int32, name="gender")
      gender_emb = layers.Embedding(input_dim=2, output_dim=16)(inp_gender)
      gender_emb = layers.Flatten()(gender_emb)
      features = layers.Concatenate()([features, gender_emb])
      inputs["gender"] = inp_gender

   x = layers.Dense(max(64, dense_units * 2), activation="relu")(features)
   output_age = layers.Dense(1, activation="linear", name="age_months")(x)

   model = Model(inputs=inputs, outputs=output_age, name=name)
   
   return model



def ROI_CNN_head (x: tf.Tensor, channels: Sequence[int], dense_units: int) -> tf.Tensor:
   """
   A CNN head for a ROI crop.

   Args:
      x (tf.Tensor): ROI image [B, H, W, 1] float32.
      channels (Sequence[int]): e.g. [32, 64]
      dense_units (int): Units in the per-ROI embedding.

   Returns:
      tf.Tensor: ROI feature embedding [B, dense_units].
   """
   
   for ch in channels:
      x = layers.Conv2D(ch, 3, padding='same', use_bias=False)(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
      x = layers.MaxPool2D(2)(x)
   
   x = layers.GlobalAveragePooling2D()(x)
   x = layers.Dense(dense_units, activation='relu')(x)
   return x
