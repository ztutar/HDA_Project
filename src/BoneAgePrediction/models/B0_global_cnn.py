
#TODO: add docstring explanation for this module. write in details and explain each function and class


from typing import Sequence, Tuple
import tensorflow as tf
from keras import layers, Model
from BoneAgePrediction.utils.logger import get_logger

logger = get_logger(__name__)

def build_GlobalCNN(input_shape: Tuple[int, int, int] = (512, 512, 1),
                     channels: Sequence[int] = (32, 64, 128),
                     dense_units: int = 64,
                     use_gender: bool = False) -> Model:
   """
   Builds a global CNN model for feature extraction.
   
   Architecture:
      [ (Conv-BN-ReLU) x2 -> MaxPool ] * len(channels)
      -> GlobalAveragePooling -> Dense(dense_units) -> Dense(1)
   
   Args:
      input_shape (Tuple[int, int, int], optional): Shape of the input image. Defaults to (512, 512, 1).
      channels (Sequence[int], optional): Number of channels for each block. Defaults to (32, 64, 128).
      dense_units (int, optional): Number of units in the dense layer before the output. Defaults to 64.
   
   Returns:
      Model: Compiled model mapping image -> age (months).
                     Inputs:  image: float32 [B,H,W,1]
                     Outputs: age  : float32 [B,1]
   """
   input_image = layers.Input(shape=input_shape, dtype=tf.float32, name="image")
   
   x = input_image
   for ch in channels:
      # two conv layers per block, then downsample
      x = layers.Conv2D(filters=ch, kernel_size=3, strides=1, padding="same", use_bias=False)(x) # [B,H,W,ch]
      x = layers.BatchNormalization()(x) # [B,H,W,ch]
      x = layers.ReLU()(x) # [B,H,W,ch]

      x = layers.Conv2D(filters=ch, kernel_size=3, strides=1, padding="same", use_bias=False)(x) # [B,H,W,ch]
      x = layers.BatchNormalization()(x) # [B,H,W,ch]
      x = layers.ReLU()(x) # [B,H,W,ch]
      
      x = layers.MaxPooling2D(pool_size=2)(x) # [B,H/2,W/2,ch]
      
   # global average pooling for feature aggregation
   x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x) # [B,ch]
   
   # gender
   if use_gender:
      input_gender = layers.Input(shape=(), dtype=tf.int32, name="gender") # [B,]
      gender_embed = layers.Embedding(input_dim=2, output_dim=8)(input_gender) # [B,8]
      gender_embed = layers.Flatten(name="gender_emb")(gender_embed) # [B,8]
      x = layers.Concatenate(name="features_with_gender")([x, gender_embed]) # [B,ch+8]
      inputs = [input_image, input_gender]
      name = "B0_Global_CNN_with_gender"
      logger.info("Building GlobalCNN model with gender input.")
   else:
      inputs = [input_image]
      name = "B0_Global_CNN"
      logger.info("Building GlobalCNN model w/o gender input.")
   
   x = layers.Dropout(rate=0.2)(x) # [B,ch(+8)]
   x = layers.Dense(units=dense_units, activation='relu')(x) # [B,dense_units]
   output_age = layers.Dense(units=1, activation='linear', name='age_months')(x) # [B,1]
   
   model = Model(inputs=inputs, outputs=output_age, name=name)
   return model
