
#TODO: add docstring explanation for this module. write in details and explain each function and class


from typing import Sequence, Tuple
import tensorflow as tf
from keras import layers, Model
from BoneAgePrediction.utils.logger import get_logger

logger = get_logger(__name__)

def build_GlobalCNN(
   input_shape: Tuple[int, int, int] = (512, 512, 1),
   channels: Sequence[int] = (32, 64, 128),
   dense_units: int = 64,
   dropout_rate: float = 0.2,
   use_gender: bool = False
) -> Model:
   """
   Build the B0 Global CNN model for bone age prediction from full images.
   Args:
      input_shape (Tuple[int, int, int], optional): Input image shape. Defaults to (512, 512, 1).
      channels (Sequence[int], optional): Conv channels per block. Defaults to (32, 64, 128).
      dense_units (int, optional): Dense units before regression. Defaults to 64.
      use_gender (bool, optional): If True, include gender embedding. Defaults to False.
   Returns:
      Model: Compiled model mapping images -> age (months).
                  Inputs: {"image": [B,H,W,1], "gender": [B,]}
                  Output: [B,1] age (months)
   """
   input_image = layers.Input(shape=input_shape, dtype=tf.float32, name="image")
   
   x = input_image
   for i, ch in enumerate(channels):
      # two conv layers per block, then downsample
      x = layers.Conv2D(
         filters=ch, 
         kernel_size=3, 
         padding="same", 
         use_bias=False,
         name=f"conv_block_{i+1}_conv1"
      )(x) # [B,H,W,ch]
      x = layers.BatchNormalization(name=f"conv_block_{i+1}_bn1")(x) # [B,H,W,ch]
      x = layers.ReLU(name=f"conv_block_{i+1}_relu1")(x) # [B,H,W,ch]

      x = layers.Conv2D(
         filters=ch, 
         kernel_size=3, 
         padding="same", 
         use_bias=False,
         name=f"conv_block_{i+1}_conv2"
      )(x) # [B,H,W,ch]
      x = layers.BatchNormalization(name=f"conv_block_{i+1}_bn2")(x) # [B,H,W,ch]
      x = layers.ReLU(name=f"conv_block_{i+1}_relu2")(x) # [B,H,W,ch]
      
      x = layers.MaxPooling2D(pool_size=2, name=f"conv_block_{i+1}_pool")(x) # [B,H/2,W/2,ch]
      
   # global average pooling for feature aggregation
   x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x) # [B,ch]
   
   # gender
   if use_gender:
      input_gender = layers.Input(shape=(), dtype=tf.int32, name="gender") # [B,]
      gender_embed = layers.Embedding(input_dim=2, output_dim=8, name="gender_embed")(input_gender) # [B,8]
      gender_embed = layers.Flatten(name="gender_embed_flat")(gender_embed) # [B,8]
      x = layers.Concatenate(name="features_with_gender")([x, gender_embed]) # [B,ch+8]
      inputs = [input_image, input_gender]
      name = "Global_CNN_with_gender"
      logger.info("Building GlobalCNN model with gender input.")
   else:
      inputs = [input_image]
      name = "Global_CNN"
      logger.info("Building GlobalCNN model w/o gender input.")
   
   if dropout_rate > 0:
      x = layers.Dropout(rate=dropout_rate, name="dropout")(x) # [B,ch(+8)]
   x = layers.Dense(
      units=dense_units, 
      activation='relu',
      name='dense'
   )(x) # [B,dense_units]
   output_age = layers.Dense(units=1, activation='linear', name='age_months', dtype=tf.float32)(x) # [B,1]
   
   model = Model(inputs=inputs, outputs=output_age, name=name)
   return model
