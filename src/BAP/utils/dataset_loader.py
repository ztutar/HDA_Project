"""
This module provides utilities for loading and preprocessing the RSNA bone age dataset.

It includes functions to download the dataset, create TensorFlow datasets, and apply image preprocessing.
"""

import cv2
from typing import Dict
import numpy as np
import tensorflow as tf
import kagglehub
from pathlib import Path
import pandas as pd
#from BAP.utils.logger import get_logger   

#logger = get_logger(__name__)


def get_rsna_dataset(force_download=False) -> dict[str, Path]:
   """
   Download and return paths to the RSNA bone age dataset splits.

   Parameters
   ----------
   force_download : bool
      Whether to force re-download of the dataset.

   Returns
   -------
   dict[str, Path]
      Dictionary with paths to root, train, validation, and test directories.
   """
   root = Path(kagglehub.dataset_download("ipythonx/rsna-bone-age", force_download=force_download))
   return {
      "root": root,
      "train": root / "RSNA_train/images",
      "validation": root / "RSNA_val/images",
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
   """
   Create a TensorFlow dataset from image directory and metadata.

   Parameters
   ----------
   image_dir : Path
      Path to the directory containing images.
   metadata : pd.DataFrame
      DataFrame with image metadata.
   image_size : int
      Target image size.
   clahe : bool
      Whether to apply CLAHE.
   augment : bool
      Whether to apply data augmentation.

   Returns
   -------
   tf.data.Dataset
      The prepared TensorFlow dataset.
   """
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
      """
      Load and preprocess a single example.

      Parameters
      ----------
      image_id : tf.Tensor
         Image ID tensor.
      age : tf.Tensor
         Age tensor.
      gender : tf.Tensor
         Gender tensor.

      Returns
      -------  
      tuple
         A tuple of (features dict, age).
      """
      image_path = tf.strings.join([base_dir, "/", image_id, ".png"])
      image = load_image_grayscale(image_path)
      image = tf.image.resize(image, (image_size, image_size), antialias=True)
      if clahe:
         image = apply_clahe(image)
      if augment:
         image = _augment_image(image)
         
      #image_viz = tf.clip_by_value(image, 0.0, 1.0)
      image_viz = image
      image = _zscore_norm(image)

      features = {
         "image_id": image_id, 
         "image": image, 
         "image_viz": image_viz,
         "gender": gender,
      }
      return features, age

   return dataset.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------
def load_image_grayscale(image_path: tf.Tensor) -> tf.Tensor:
   """
   Load a grayscale image from file path.

   Parameters
   ----------
   image_path : tf.Tensor
      Path to the image file.

   Returns
   -------
   tf.Tensor
      Loaded grayscale image tensor.
   """
   image = tf.io.read_file(image_path)
   image = tf.image.decode_png(image, channels=1)  # Grayscale
   image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
   return image # [H,W,1], float32 in [0,1]

def load_image_original(image_path: tf.Tensor) -> tf.Tensor:
   """
   Load an image with original channels from file path.

   Parameters
   ----------
   image_path : tf.Tensor
      Path to the image file.

   Returns
   -------
   tf.Tensor
      Loaded image tensor with original channels.
   """
   image = tf.io.read_file(image_path)
   image = tf.image.decode_png(image, channels=3)  # Original channels
   image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
   return image # [H,W,3], float32 in [0,1]

def apply_clahe(image: tf.Tensor) -> tf.Tensor:
   """
   Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

   Parameters
   ----------
   image : tf.Tensor
      Input image tensor.

   Returns
   -------
   tf.Tensor
      Image with CLAHE applied.
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
   Apply z-score normalization to an image.

   Parameters
   ----------
   image : tf.Tensor
      Input image tensor.
   eps : float
      Small epsilon for numerical stability.

   Returns
   -------
   tf.Tensor
      Normalized image tensor.
   """
   mean, variance = tf.nn.moments(image, axes=[0, 1], keepdims=True)
   stddev = tf.sqrt(variance)
   return (image - mean) / (stddev + eps)

def _augment_image(image: tf.Tensor) -> tf.Tensor:
   """
   Apply data augmentation to an image.

   Parameters
   ----------
   image : tf.Tensor
      Input image tensor.

   Returns
   -------
   tf.Tensor
      Augmented image tensor.
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
   Rotate an image by a given angle.

   Parameters
   ----------
   image : tf.Tensor
      Input image tensor.
   angle_rad : tf.Tensor
      Rotation angle in radians.

   Returns
   -------
   tf.Tensor
      Rotated image tensor.
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
