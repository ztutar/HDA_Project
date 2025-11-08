
"""Utility functions for loading and preparing the hand bone age dataset.

This module brings together helpers for reading metadata from CSV files,
linking those records to image files ...
"""

import os
import glob
import csv
import cv2
from typing import Dict, Tuple, List, Optional
import numpy as np
import tensorflow as tf
import kagglehub
from pathlib import Path
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
   image_dir: str,
   metadata: Optional[str] = None,
   grayscale: bool = True,
   resize: bool = True,
   image_size: int = 512,
   clahe: bool = False,
   augment: bool = False,
) -> tf.data.Dataset:
   """
   Loads a TensorFlow dataset for bone age images and labels.
   Args:
   Returns:
      tf.data.Dataset: A TensorFlow dataset yielding (image, label) pairs.  
   """
   images = []
   ages = []
   genders = []
   for _, row in metadata.iterrows():
      image_id = row["Image ID"]
      image_path = os.path.join(image_dir, f"{image_id}.png")
      try:
         if grayscale:
            image = load_image_grayscale(image_path)
         if resize:
            image = tf.image.resize(image, (image_size, image_size), antialias=True)
         if clahe:
            image = apply_clahe(image)  # CLAHE preprocessing
         if augment:
            image = _augment_image(image)  # data augmentation
         image = _zscore_norm(image)  # z-score normalization
         images.append(image)
         ages.append(row["Bone Age (months)"])
         genders.append(row["male"])
      except FileNotFoundError:
         #logger.info(f"Skipping {image_path} due to missing file.")
         print(f"Skipping {image_path} due to missing file.")
         continue
      
   image_ds = tf.data.Dataset.from_tensor_slices(tf.stack(images))
   age_ds = tf.data.Dataset.from_tensor_slices(tf.constant(ages, dtype=tf.float32))
   gender_ds = tf.data.Dataset.from_tensor_slices(tf.constant(genders, dtype=tf.int32))
   dataset = tf.data.Dataset.zip((image_ds, age_ds, gender_ds))
   return dataset

# ---------- ROI Dataset loader (load saved crops) ----------
def make_roi_dataset(
   image_dir: str,
   roi_path: str,
   metadata: Optional[str] = None,
   split: str = "train",
   batch_size: int = 16,
) -> tf.data.Dataset:
   """
   Build a tf.data pipeline yielding paired ROI crops and labels:
   ({"carpal": img, "metaph": img, "gender": int32}, age_months)

   Args:
      image_dir (str): Root directory containing the split image folders.
      metadata (str | None): Directory containing split CSV metadata; defaults to image_dir.
      roi_path (str): Base dir with saved crops: {roi_root}/{split}/{carpal,metaph}/*.png
      split (str): 'train' | 'validation' | 'test'
      batch_size (int): Batch size.

   Returns:
      tf.data.Dataset: Batches of paired crops and labels.
   """
   split = {"train": "train", "val": "validation", "validation":"validation", "test": "test"}[split.lower()]
   carpal_dir = os.path.join(roi_path, split, "carpal")
   metaph_dir = os.path.join(roi_path, split, "metaph")
   
   # Expect original labels CSV from metadata (or image_dir fallback)
   csv_root = metadata or image_dir
   labels_csv = os.path.join(csv_root, f"{split}.csv")
   rows = read_csv_labels(labels_csv)
   
   # Build id -> path maps
   carpal_map = id_to_path(carpal_dir)
   metaph_map = id_to_path(metaph_dir)
   
   carpal_paths, metaph_paths, genders, ages, ids = [], [], [], [], []
   missing_images = []
   missing_coords = []
   for r in rows:
      image_id = r["image_id"]
      carp = carpal_map.get(image_id, None)
      metp = metaph_map.get(image_id, None)
      
      if carp is None or metp is None:
         missing_images.append(image_id)
         continue

      carpal_paths.append(carp)
      metaph_paths.append(metp)
      genders.append(r["male"])
      ages.append(r["age_months"])
      ids.append(image_id)
      
   if missing_images:
      logger.info("%d ROI crops missing for %s split (examples: %s)", len(missing_images), split, missing_images[:5])
   if missing_coords:
      logger.info("%d ROI coords missing for %s split (examples: %s)", len(missing_coords), split, missing_coords[:5])
      
   if not carpal_paths:
      raise FileNotFoundError(
         f"No ROI crops found in {carpal_dir}. Regenerate crops before training the ROI model."
      )
   
   carpal_path_ds = tf.data.Dataset.from_tensor_slices(tf.constant(carpal_paths, dtype=tf.string))
   metaph_path_ds = tf.data.Dataset.from_tensor_slices(tf.constant(metaph_paths, dtype=tf.string))
   gender_ds = tf.data.Dataset.from_tensor_slices(np.array(genders, dtype=np.int32))
   age_ds = tf.data.Dataset.from_tensor_slices(np.array(ages, dtype=np.float32))
   id_ds = tf.data.Dataset.from_tensor_slices(np.array(ids, dtype=np.str_))
   dataset = tf.data.Dataset.zip((carpal_path_ds, metaph_path_ds, gender_ds, age_ds, id_ds))
      
   def _load_pair(
      carpal_path: tf.Tensor,
      metaph_path: tf.Tensor,
      gender: tf.Tensor,
      age: tf.Tensor,
      img_id: tf.Tensor,
   ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
      """
      Loads and preprocesses a single image and its label.

      Args:
         carpal_path (tf.Tensor): tf.string file path to cropped carpal image.
         metaph_path (tf.Tensor): tf.string file path to cropped metaphalange image.
         gender (tf.Tensor): tf.int32 scalar {0,1}.
         age (tf.Tensor): tf.float32 scalar (months).
         img_id (tf.Tensor): tf.string scalar, numeric ID from CSV (e.g., "4516").

      Returns:
         (features, label):
            features = {
               "carpal":     tf.float32 [H,W,1],
               "metaph":     tf.float32 [H,W,1],
               "gender":     tf.int32   [],
               "image_id":   tf.string []
            }
            label = tf.float32 []  # age in months
            Note: ROI boxes are (-1,-1,-1,-1) when coordinates are unavailable.
      """
      c_img = _zscore_norm(load_image_grayscale(carpal_path))  # [H,W,1], float32 in [0,1], z-score normalized
      m_img = _zscore_norm(load_image_grayscale(metaph_path))
         
      features = {
         "carpal": c_img, 
         "metaph": m_img,
         "gender": tf.cast(gender, tf.int32),
         "image_id": img_id
      }
      return features, tf.cast(age, tf.float32)
   
   dataset = dataset.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
   dataset = dataset.batch(batch_size, drop_remainder=False)
   dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   return dataset


def make_fusion_dataset(
   image_dir: str,
   roi_path: str,
   metadata: Optional[str] = None,
   split: str = "train",
   image_size: int = 512,
   keep_aspect_ratio: bool = True,
   batch_size: int = 16,
   clahe: bool = False,
   augment: bool = True,
) -> tf.data.Dataset:
   """
   Build a tf.data pipeline yielding full images paired with ROI crops.
   """
   split_key = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}[split.lower()]
   image_dir = os.path.join(image_dir, split_key)
   csv_root = metadata or image_dir
   labels_csv = os.path.join(csv_root, f"{split_key}.csv")
   carpal_dir = os.path.join(roi_path, split_key, "carpal")
   metaph_dir = os.path.join(roi_path, split_key, "metaph")
   logger.info("Preparing fusion dataset | split=%s", split_key)

   rows = read_csv_labels(labels_csv)
   image_map = id_to_path(image_dir)
   carpal_map = id_to_path(carpal_dir)
   metaph_map = id_to_path(metaph_dir)

   image_paths, carpal_paths, metaph_paths = [], [], []
   genders, ages, ids = [], [], []
   missing_global, missing_rois = [], []

   for row in rows:
      image_id = row["image_id"]
      image_path = image_map.get(image_id)
      carpal_path = carpal_map.get(image_id)
      metaph_path = metaph_map.get(image_id)

      if image_path is None:
         missing_global.append(image_id)
         continue
      if carpal_path is None or metaph_path is None:
         missing_rois.append(image_id)
         continue

      image_paths.append(image_path)
      carpal_paths.append(carpal_path)
      metaph_paths.append(metaph_path)
      genders.append(row["male"])
      ages.append(row["age_months"])
      ids.append(image_id)

   if missing_global:
      logger.info("%d full-hand images missing for %s split (examples: %s)", len(missing_global), split_key, missing_global[:5])
   if missing_rois:
      logger.info("%d ROI crops missing for %s split (examples: %s)", len(missing_rois), split_key, missing_rois[:5])

   if not image_paths:
      raise FileNotFoundError(
         f"No fusion samples found in {image_dir}. Ensure full images and ROI crops are prepared."
      )

   image_path_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_paths, dtype=tf.string))
   carpal_path_ds = tf.data.Dataset.from_tensor_slices(tf.constant(carpal_paths, dtype=tf.string))
   metaph_path_ds = tf.data.Dataset.from_tensor_slices(tf.constant(metaph_paths, dtype=tf.string))
   gender_ds = tf.data.Dataset.from_tensor_slices(np.array(genders, dtype=np.int32))
   age_ds = tf.data.Dataset.from_tensor_slices(np.array(ages, dtype=np.float32))
   id_ds = tf.data.Dataset.from_tensor_slices(np.array(ids, dtype=np.str_))

   dataset = tf.data.Dataset.zip((image_path_ds, carpal_path_ds, metaph_path_ds, gender_ds, age_ds, id_ds))
   is_train = split_key == "train"

   def _load_sample(
      image_path: tf.Tensor,
      carpal_path: tf.Tensor,
      metaph_path: tf.Tensor,
      gender: tf.Tensor,
      age: tf.Tensor,
      img_id: tf.Tensor,
   ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
      image = load_image_grayscale(image_path)
      if keep_aspect_ratio:
         image, _ = _resize_with_letterbox(image, image_size)
      else:
         image, _ = resize_image(image, image_size)

      if clahe:
         image = apply_clahe(image)
      if is_train and augment:
         image = _augment_image(image)

      image_viz = tf.clip_by_value(image, 0.0, 1.0)
      image = _zscore_norm(image)

      carpal_img = _zscore_norm(load_image_grayscale(carpal_path))
      metaph_img = _zscore_norm(load_image_grayscale(metaph_path))

      features = {
         "image": image,
         "image_viz": image_viz,
         "carpal": carpal_img,
         "metaph": metaph_img,
         "gender": tf.cast(gender, tf.int32),
         "image_id": img_id,
      }
      return features, tf.cast(age, tf.float32)

   dataset = dataset.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)
   dataset = dataset.batch(batch_size, drop_remainder=False)
   dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   return dataset

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

# ---------- CSV and file indexing helpers ----------
def read_csv_labels(csv_path: str) -> List[Dict]:
   """
   Reads a CSV file with header: 'Image ID,male,Bone Age (months)'.
   Args:
      csv_path (str): Path to the CSV file.
   Returns:
      List[Dict]: A list of dicts: {'image_id': '4360', 'male': 1, 'age_months': 168.93}
   """
   rows = []
   with open(csv_path, newline="") as csvfile:
      reader = csv.DictReader(csvfile)
      for r in reader:
         image_id = str(r['Image ID']).strip()
         male_raw = str(r["male"]).strip().upper()
         male = 1 if male_raw in ("1", "TRUE", "T", "YES", "Y") else 0
         age_months = float(r["Bone Age (months)"])
         rows.append({"image_id": image_id, "male": male, "age_months": age_months})
   logger.info("Loaded %d rows from %s", len(rows), csv_path)
   return rows

def id_to_path(image_dir: str) -> Dict[str, str]:
   """
   Builds a mapping from image ID to full file path.
   'id' is the stem (filename without extension). e.g., '4360' -> '/.../4360.png'
   Args:
      image_dir (str): Directory containing images.
   Returns:
      id_path_map (Dict[str, str]): Mapping from image ID to file path.
   """
   pattern = os.path.join(image_dir, "*.png")
   files = glob.glob(pattern)
   id_path_map = {}
   for f in files:
      base = os.path.basename(f)
      image_id = os.path.splitext(base)[0]
      id_path_map[image_id] = f
   logger.debug("Indexed %d images in %s", len(id_path_map), image_dir)
   return id_path_map

# ---------- Image processing helpers ----------
def load_image_grayscale(image_path: tf.Tensor) -> tf.Tensor:
   image = tf.io.read_file(image_path)
   image = tf.image.decode_png(image, channels=1)  # Grayscale
   image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
   return image # [H,W,1], float32 in [0,1]

def load_image_original(image_path: tf.Tensor) -> tf.Tensor:
   image = tf.io.read_file(image_path)
   image = tf.image.decode_png(image, channels=3)  # Original channels
   image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
   return image # [H,W,1], float32 in [0,1]

def resize_image(img: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, Dict]:
   img = tf.image.resize(img, (image_size, image_size), antialias=True)
   return img

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
