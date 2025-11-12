
"""Utility functions for loading and preparing the hand bone age dataset.

This module brings together helpers for reading metadata from CSV files,
linking those records to image files ...
"""

#import os
#import glob
#import csv
import cv2
from typing import Dict, Tuple, List, Optional
import numpy as np
import tensorflow as tf
import kagglehub
from pathlib import Path
import pandas as pd
#from BAP.utils.logger import get_logger   

#logger = get_logger(__name__)

def get_rsna_dataset(force_download=False) -> dict[str, Path]:
   root = Path(kagglehub.dataset_download("ipythonx/rsna-bone-age", force_download=force_download))
   return {
      "root": root,
      "train": root / "RSNA_train/images",
      "val": root / "RSNA_val/images",
      "test": root / "RSNA_test/images",
   }
   
# ------------------------------------
#  Dataset loader 
# ------------------------------------
def make_dataset(
   image_dir: Path,
   metadata: pd.DataFrame,
   image_size: int = 512,
   clahe: bool = False,
   augment: bool = False,
) -> tf.data.Dataset:

   image_ids = metadata["Image ID"].astype(str).tolist()
   ages = metadata["Bone Age (months)"].astype(np.float32).tolist()
   male_column = metadata.get("male")
   genders = male_column.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES", "Y"]).tolist()
   
   base_dir = tf.constant(str(image_dir))

   dataset = tf.data.Dataset.from_tensor_slices((
      tf.constant(image_ids),
      tf.constant(ages, dtype=tf.float32),
      tf.constant(genders, dtype=tf.bool),
   ))

   def _load_example(image_id: tf.Tensor, age: tf.Tensor, gender: tf.Tensor):
      image_path = tf.strings.join([base_dir, "/", image_id, ".png"])
      image = load_image_grayscale(image_path)
      image = tf.image.resize(image, (image_size, image_size), antialias=True)
      if clahe:
         image = apply_clahe(image)
      if augment:
         image = _augment_image(image)
         
      image = tf.clip_by_value(image, 0.0, 1.0)
      features = {
         "image_id": image_id, 
         "image": image, 
         "gender": gender,
      }
      return features, age

   return dataset.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)

# ---------- ROI Dataset loader (load saved crops) ----------
def make_roi_dataset(
   roi_dir: Dict,
   metadata: pd.DataFrame,
) -> tf.data.Dataset:
   
   image_ids = metadata["Image ID"].astype(str).tolist()
   ages = metadata["Bone Age (months)"].astype(np.float32).tolist()
   male_column = metadata.get("male")
   genders = male_column.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES", "Y"]).tolist()
   
   carpal_base = tf.constant(str(roi_dir["carpal"]))
   metaph_base = tf.constant(str(roi_dir["metaph"]))

   dataset = tf.data.Dataset.from_tensor_slices((
      tf.constant(image_ids),
      tf.constant(ages, dtype=tf.float32),
      tf.constant(genders, dtype=tf.bool),
   ))

   def _load_roi(image_id: tf.Tensor, age: tf.Tensor, gender: tf.Tensor):
      carpal_path = tf.strings.join([carpal_base, "/", image_id, ".png"])
      metaph_path = tf.strings.join([metaph_base, "/", image_id, ".png"])
      #carpal = _zscore_norm(load_image_grayscale(carpal_path))
      #metaph = _zscore_norm(load_image_grayscale(metaph_path))
      carpal = tf.clip_by_value(load_image_grayscale(carpal_path), 0.0, 1.0)
      metaph = tf.clip_by_value(load_image_grayscale(metaph_path), 0.0, 1.0)
      
      features = {
         "image_id": image_id,
         "carpal": carpal,
         "metaph": metaph,
         "gender": gender,
      }
      return features, age
   
   dataset = dataset.map(_load_roi, num_parallel_calls=tf.data.AUTOTUNE)
   return dataset

# ---------- Fusion Dataset loader ----------
def make_fusion_dataset(
   image_dir: Path,
   roi_dir: Dict,
   metadata: pd.DataFrame,
   image_size: int = 512,
   clahe: bool = False,
   augment: bool = False,
) -> tf.data.Dataset:
   """
   Build a tf.data pipeline yielding full images paired with ROI crops.
   """
   
   image_ids = metadata["Image ID"].astype(str).tolist()
   ages = metadata["Bone Age (months)"].astype(np.float32).tolist()
   male_column = metadata.get("male")
   genders = male_column.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES", "Y"]).tolist()

   image_base = tf.constant(str(image_dir))
   carpal_base = tf.constant(str(roi_dir["carpal"]))
   metaph_base = tf.constant(str(roi_dir["metaph"]))

   dataset = tf.data.Dataset.from_tensor_slices((
      tf.constant(image_ids),
      tf.constant(ages, dtype=tf.float32),
      tf.constant(genders, dtype=tf.bool),
   ))

   def _load_fusion(image_id: tf.Tensor, age: tf.Tensor, gender: tf.Tensor):
      image_path = tf.strings.join([image_base, "/", image_id, ".png"])
      carpal_path = tf.strings.join([carpal_base, "/", image_id, ".png"])
      metaph_path = tf.strings.join([metaph_base, "/", image_id, ".png"])

      image = load_image_grayscale(image_path)
      image = tf.image.resize(image, (image_size, image_size), antialias=True)
      if clahe:
         image = apply_clahe(image)
      if augment:
         image = _augment_image(image)
      #image = _zscore_norm(image)
      image = tf.clip_by_value(image, 0.0, 1.0)
      
      carpal = tf.clip_by_value(load_image_grayscale(carpal_path), 0.0, 1.0)
      metaph = tf.clip_by_value(load_image_grayscale(metaph_path), 0.0, 1.0)

      features = {
         "image_id": image_id,
         "image": image,
         "carpal": carpal,
         "metaph": metaph,
         "gender": gender,
      }
      return features, age

   return dataset.map(_load_fusion, num_parallel_calls=tf.data.AUTOTUNE)

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------


def load_image_grayscale(image_path: tf.Tensor) -> tf.Tensor:
   image = tf.io.read_file(image_path)
   image = tf.image.decode_png(image, channels=1)  # Grayscale
   image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
   return image # [H,W,1], float32 in [0,1]

def load_image_original(image_path: tf.Tensor) -> tf.Tensor:
   image = tf.io.read_file(image_path)
   image = tf.image.decode_png(image, channels=3)  # Original channels
   image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
   return image # [H,W,3], float32 in [0,1]

def apply_clahe(image: tf.Tensor) -> tf.Tensor:
   """
   Applies CLAHE to a grayscale image tensor.
   image: float32 [0,1], shape [H,W,1] -> apply CLAHE in numpy (uint8) -> back to float32 [0,1]
   Args:
      image (tf.Tensor): Input image tensor of shape (H, W, 1) with dtype float32 in [0, 1].
   Returns:
      tf.Tensor: CLAHE processed image tensor of shape (H, W, 1) with dtype float32 in [0, 1].
   """
   image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)  # [H,W,1], uint8 in [0,255]
   image_uint8 = tf.squeeze(image_uint8, axis=-1)  # [H,W], uint8
   
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # create CLAHE object
   def _clahe_apply(img_np: np.ndarray) -> np.ndarray:
      return clahe.apply(img_np)
   
   image_clahe = tf.numpy_function(func=_clahe_apply, inp=[image_uint8], Tout=tf.uint8) # [H,W], uint8
   image_clahe = tf.expand_dims(image_clahe, axis=-1)  # [H,W,1], uint8
   image_clahe = tf.image.convert_image_dtype(image_clahe, tf.float32)  # [H,W,1], float32 in [0,1]
   image_clahe = tf.ensure_shape(image_clahe, image.shape)
   return image_clahe  

def _zscore_norm(image: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
   """
   Applies z-score normalization to an image tensor.
   Args:
      image (tf.Tensor): Input image tensor.
      eps (float): Small epsilon value to avoid division by zero.
   Returns:
      tf.Tensor: Z-score normalized image tensor.
   """
   mean, variance = tf.nn.moments(image, axes=[0, 1], keepdims=True)
   stddev = tf.sqrt(variance)
   return (image - mean) / (stddev + eps)

def _augment_image(image: tf.Tensor) -> tf.Tensor:
   """
   Applies random data augmentations to an image tensor.
   Augmentations include random flips and rotations.
   Args:
      image (tf.Tensor): Input image tensor of shape (H, W, C).  
   Returns:
      tf.Tensor: Augmented image tensor.
   """
   # random flips
   image = tf.image.random_flip_left_right(image)
   # random rotations (~±2 degrees)
   angle = tf.random.uniform([], minval=-2.0, maxval=2.0) * (np.pi / 180.0)  # Convert degrees to radians
   image = _image_rotate(image, angle)
   # brightness and contrast jitter
   image = tf.image.random_brightness(image, max_delta=0.02)
   image = tf.image.random_contrast(image, lower=0.98, upper=1.02)
   return image

def _image_rotate(image: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
   """
   Rotates an image by a given angle in radians using affine_grid + sampling implemented with tf.raw_ops.
   Steps:
   1) Build the rotation matrix.
   2) Use tf.raw_ops.ImageProjectiveTransformV3 to apply the transformation.
   3) Fill the empty pixels using "REFLECT" mode.
   Args:
      image (tf.Tensor): Input image tensor of shape (H, W, C).
      angle_rad (tf.Tensor): Rotation angle in radians (scalar).
   Returns:
      tf.Tensor: Rotated image tensor of shape (H, W, C).
   """
   # center-based rotation
   h = tf.cast(tf.shape(image)[0], tf.float32)
   w = tf.cast(tf.shape(image)[1], tf.float32)
   center = tf.stack([(w - 1.0) / 2.0, (h - 1.0) / 2.0]) 
   cos_angle = tf.cos(angle_rad)
   sin_angle = tf.sin(angle_rad)    
   # build the rotation matrix
   transform = tf.stack([
      cos_angle, 
      -sin_angle, 
      (1.0 - cos_angle) * center[0] + sin_angle * center[1],
      sin_angle,  
      cos_angle, 
      (1.0 - cos_angle) * center[1] - sin_angle * center[0],
      tf.constant(0.0, dtype=tf.float32),
      tf.constant(0.0, dtype=tf.float32),
   ])
   transform = tf.reshape(transform, [1, 8])
   
   # apply the transformation
   image = tf.expand_dims(image, axis=0)  # [1,H,W,1]
   out = tf.raw_ops.ImageProjectiveTransformV3(
      images=image,
      transforms=transform,
      output_shape=tf.shape(image)[1:3],
      interpolation="BILINEAR",
      fill_mode="REFLECT",
      fill_value=0.0,
   )
   out = tf.squeeze(out, axis=0)  # [H,W,1]
   return out
# -----------------------------------------
