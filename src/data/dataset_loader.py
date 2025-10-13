
"""Utility functions for loading and preparing the hand bone age dataset.

This module brings together helpers for reading metadata from CSV files,
linking those records to image files on disk, and preparing image tensors for
TensorFlow workflows.

Main components:
   -  CSV helpers (`read_csv_labels`) that parse the bone age labels file and
      normalize key fields such as the gender flag and bone age measurement.
   -  File indexing helpers (`build_id_to_path`) that create fast lookups from
      numeric study identifiers to the corresponding `.png` image path.
   -  Image preprocessing routines (`read_image_grayscale`, optional CLAHE
      helpers) that read images, convert them to grayscale tensors, and apply
      optional contrast limited adaptive histogram equalization (CLAHE) when 
      OpenCV is available.
"""

import os
import glob
import csv
from typing import Dict, Tuple, List
import numpy as np
import tensorflow as tf
import logging
try:
   # Try relative import when running as a package
   from utils.logger import get_logger  # type: ignore
except Exception:  # pragma: no cover - fallback in ad-hoc runs
   try:
      from src.utils.logger import get_logger  # type: ignore
   except Exception:
      get_logger = logging.getLogger  # type: ignore

try:
   import cv2 # used only if CLAHE = True
   _HAS_CV2 = True
except Exception:
   _HAS_CV2 = False
   
# ---------- CSV reading helpers ----------
def read_csv_labels(csv_path: str) -> List[Dict]:
   """
   Reads a CSV file with header: 'Image ID,male,Bone Age (months)'.
   Args:
      csv_path (str): Path to the CSV file.
   Returns:
      List[Dict]: A list of dicts: {'image_id': '4360', 'male': 1, 'age_months': 168.93}
   """
   logger = get_logger(__name__)
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

def build_id_to_path(image_dir: str) -> Dict[str, str]:
   """
   Builds a mapping from image ID to full file path.
   'id' is the stem (filename without extension). e.g., '4360' -> '/.../4360.png'
   Args:
      image_dir (str): Directory containing images.
   Returns:
      Dict[str, str]: Mapping from image ID to file path.
   """
   pattern = os.path.join(image_dir, "*.png")
   logger = get_logger(__name__)
   files = glob.glob(pattern)
   id_to_path = {}
   for f in files:
      base = os.path.basename(f)
      image_id = os.path.splitext(base)[0]
      id_to_path[image_id] = f
   logger.debug("Indexed %d images in %s", len(id_to_path), image_dir)
   return id_to_path

# ---------- Image processing helpers ----------
def read_image_grayscale(image_path: tf.Tensor) -> tf.Tensor:
   """
   Reads and preprocesses a grayscale image from a file path.
   Args:
      image_path (tf.Tensor): Path to the image file.
   Returns:
      tf.Tensor: Preprocessed image tensor of shape (image_size, image_size, 1).
   """
   image = tf.io.read_file(image_path)
   image = tf.image.decode_png(image, channels=1)  # Grayscale
   image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
   return image # [H,W,1], float32 in [0,1]

def resize_with_letterbox(img: tf.Tensor, target_h: int, target_w: int,
                           pad_value: float = 0.0) -> Tuple[tf.Tensor, Dict]:
   """
   Resizes an image to fit within target_h x target_w while preserving aspect ratio.
   Pads the image with pad_value to achieve exact target size.
   Args:
      img (tf.Tensor): Input image tensor of shape (H, W, C).
      target_h (int): Target height.
      target_w (int): Target width.
      pad_value (float): Value to use for padding.
   Returns:
      tf.Tensor: Resized and padded image tensor of shape (target_h, target_w, C).
      dict: Metadata including scale and offsets for potential CAM mapping.
   """
   h = tf.cast(tf.shape(img)[0], tf.float32)
   w = tf.cast(tf.shape(img)[1], tf.float32)
   scale = tf.minimum(target_h / h, target_w / w)
   new_h = tf.cast(tf.round(h * scale), tf.int32)
   new_w = tf.cast(tf.round(w * scale), tf.int32)
   img = tf.image.resize(img, (new_h, new_w), antialias=True)
   # pad to center
   pad_h = target_h - new_h
   pad_w = target_w - new_w
   off_h = pad_h // 2
   off_w = pad_w // 2
   img = tf.image.pad_to_bounding_box(img, off_h, off_w, target_h, target_w)
   if pad_h % 2 != 0 or pad_w % 2 != 0:
      # ensure exact size if odd padding
      img = tf.image.resize_with_crop_or_pad(img, target_h, target_w)
   # return img and metadata useful for mapping CAMs back, if needed
   meta = {
      "scale": scale,
      "offset_y": tf.cast(off_h, tf.float32),
      "offset_x": tf.cast(off_w, tf.float32),
      "orig_h": h, "orig_w": w
   }
   return img, meta

def resize_without_ar(img: tf.Tensor, target_h: int, target_w: int) -> Tuple[tf.Tensor, Dict]:
   """
   Resizes an image to target_h x target_w without preserving aspect ratio.
   Args:
      img (tf.Tensor): Input image tensor of shape (H, W, C).
      target_h (int): Target height.
      target_w (int): Target width.
   Returns:
      tf.Tensor: Resized image tensor of shape (target_h, target_w, C).
      dict: Metadata with None scale and zero offsets.
   """
   img = tf.image.resize(img, (target_h, target_w), antialias=True)  # may distort
   meta = {"scale": None, "offset_y": 0., "offset_x": 0., "orig_h": tf.shape(img)[0], "orig_w": tf.shape(img)[1]}
   return img, meta


def clahe_uint8(image: np.ndarray) -> np.ndarray:
   """
   Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a uint8 grayscale image.
   Args:
      image (np.ndarray): Input image as a 2D numpy array of dtype uint8.
   Returns:
      np.ndarray: CLAHE processed image as a 2D numpy array of dtype uint8.
   """
   if not _HAS_CV2:
      raise RuntimeError("OpenCV is not installed, but CLAHE=True was requested. Please install OpenCV or set CLAHE=False.")
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   return clahe.apply(image)

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
   image_clahe = tf.numpy_function(clahe_uint8, [image_uint8], tf.uint8)  # [H,W], uint8
   image_clahe = tf.expand_dims(image_clahe, axis=-1)  # [H,W,1], uint8
   image_clahe = tf.image.convert_image_dtype(image_clahe, tf.float32)  # [H,W,1], float32 in [0,1]
   return image_clahe  

def zscore_norm(image: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
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

def augment_image(image: tf.Tensor) -> tf.Tensor:
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
   # random rotations (~±7 degrees)
   angle = tf.random.uniform([], minval=-7.0, maxval=7.0) * (np.pi / 180.0)  # Convert degrees to radians
   image = image_rotate(image, angle)
   # brightness and contrast jitter
   image = tf.image.random_brightness(image, max_delta=0.05)
   image = tf.image.random_contrast(image, lower=0.95, upper=1.05)
   return image

def image_rotate(image: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
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


# ---------- Dataset loader ----------
def make_dataset(
   data_path: str,
   split: str = "train",
   target_h: int = 512,
   target_w: int = 512,
   keep_aspect_ratio=True, 
   pad_value=0.0,
   batch_size: int = 16,
   shuffle_buffer: int = 1024,
   num_workers: int = 4,
   clahe: bool = False,
   augment: bool = True,
   cache: bool = True,
) -> tf.data.Dataset:
   """
   Loads a TensorFlow dataset for bone age images and labels.
   Args:
      data_path (str): Root directory containing 'images/' and 'labels.csv'.
      split (str): Dataset split to load ('train', 'val', or 'test').
      target_h (int): Target image height.
      target_w (int): Target image width.
      keep_aspect_ratio (bool): Whether to keep aspect ratio when resizing.
      pad_value (float): Padding value when resizing with aspect ratio.
      batch_size (int): Batch size.
      shuffle_buffer (int): Buffer size for shuffling.
      num_workers (int): Number of parallel calls for data loading and processing.
      clahe (bool): Whether to apply CLAHE preprocessing.
      augment (bool): Whether to apply data augmentation (only for 'train' split).
      cache (bool): Whether to cache the dataset in memory.
   Returns:
      tf.data.Dataset: A TensorFlow dataset yielding (image, label) pairs.  
   """
   logger = get_logger(__name__)
   split = {"train": "train", "val": "validation", "validation":"validation", "test": "test"}[split.lower()]
   image_path = os.path.join(data_path, split)
   csv_path = os.path.join(data_path, f"{split}.csv")
   logger.info("Preparing dataset | split=%s | image_path=%s", split, image_path)
   
   rows = read_csv_labels(csv_path)
   id_to_path = build_id_to_path(image_path)
   
   paths, genders, ages = [], [], []
   missing = []
   for r in rows:
      image_id = r["image_id"]
      p = id_to_path.get(image_id, None)
      if p is None:
         missing.append(image_id)
         continue
      paths.append(p)
      genders.append(r["male"])
      ages.append(r["age_months"])
      
   if len(missing) > 0:
      logger.warning("%d images not found in %s. examples=%s", len(missing), image_path, missing[:5])
   logger.info("Dataset stats | files=%d | genders=%d | ages=%d", len(paths), len(genders), len(ages))
   
   path_ds = tf.data.Dataset.from_tensor_slices(np.array(paths))
   gender_ds = tf.data.Dataset.from_tensor_slices(np.array(genders, dtype=np.int32))
   age_ds = tf.data.Dataset.from_tensor_slices(np.array(ages, dtype=np.float32))
   dataset = tf.data.Dataset.zip((path_ds, gender_ds, age_ds))
   
   is_train = split == "train"
   if is_train:
      dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
   
   def load_and_preprocess(path: tf.Tensor, gender: tf.Tensor, age: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
      """ Loads and preprocesses a single image and its label."""
      image = read_image_grayscale(path)  # [H,W,1], float32 in [0,1]
      
      if keep_aspect_ratio:
         image, _ = resize_with_letterbox(image, target_h, target_w, pad_value=pad_value)
      else:
         image, _ = resize_without_ar(image, target_h, target_w)
      
      if clahe:
         image = apply_clahe(image)  # CLAHE preprocessing
      image = zscore_norm(image)  # z-score normalization
      
      if is_train and augment:
         image = augment_image(image)  # data augmentation
         
      features = {"image": image, "gender": tf.cast(gender, tf.int32)}
      return features, tf.cast(age, tf.float32)
   
   dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
   if cache:
      dataset = dataset.cache()
   dataset = dataset.batch(batch_size, drop_remainder=False)
   dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   return dataset
