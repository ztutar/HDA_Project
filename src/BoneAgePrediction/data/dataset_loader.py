
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
   from BoneAgePrediction.utils.logger import get_logger  
except ImportError:  #fallback when package not installed
   get_logger = logging.getLogger  

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
      id_to_path (Dict[str, str]): Mapping from image ID to file path.
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
   
   paths, genders, ages, ids = [], [], [], []
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
      ids.append(image_id)
      
   if len(missing) > 0:
      logger.warning("%d images not found in %s. examples=%s", len(missing), image_path, missing[:5])
   logger.info("Dataset stats | files=%d | genders=%d | ages=%d", len(paths), len(genders), len(ages))
   
   path_ds = tf.data.Dataset.from_tensor_slices(np.array(paths))
   gender_ds = tf.data.Dataset.from_tensor_slices(np.array(genders, dtype=np.int32))
   age_ds = tf.data.Dataset.from_tensor_slices(np.array(ages, dtype=np.float32))
   id_ds = tf.data.Dataset.from_tensor_slices(np.array(ids, dtype=np.str_))
   dataset = tf.data.Dataset.zip((path_ds, gender_ds, age_ds, id_ds))
   options = tf.data.Options()
   options.experimental_deterministic = True
   dataset = dataset.with_options(options)
      
   def _load_and_preprocess(path: tf.Tensor, gender: tf.Tensor, age: tf.Tensor, img_id: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
      """
      Loads and preprocesses a single image and its label.

      Args:
         path (tf.Tensor): tf.string file path to image.
         gender (tf.Tensor): tf.int32 scalar {0,1}.
         age (tf.Tensor): tf.float32 scalar (months).
         img_id (tf.Tensor): tf.string scalar, numeric ID from CSV (e.g., "4516").

      Returns:
         (features, label):
            features = {
               "image":  tf.float32 [H,W,1],
               "gender": tf.int32   [],
               "image_id": tf.string []
            }
            label = tf.float32 []  # age in months
      """
      image = read_image_grayscale(path)  # [H,W,1], float32 in [0,1]
      
      if keep_aspect_ratio:
         image, _ = resize_with_letterbox(image, target_h, target_w, pad_value=pad_value)
      else:
         image, _ = resize_without_ar(image, target_h, target_w)
      
      if clahe:
         image = apply_clahe(image)  # CLAHE preprocessing
      image = zscore_norm(image)  # z-score normalization

      is_train = split == "train"
      if is_train and augment:
         image = augment_image(image)  # data augmentation
         
      features = {
         "image": image, 
         "gender": tf.cast(gender, tf.int32),
         "image_id": img_id
      }
      return features, tf.cast(age, tf.float32)
   
   dataset = dataset.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
   if cache:
      dataset = dataset.cache()
   dataset = dataset.batch(batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
   dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   return dataset

# ---------- ROI Dataset loader (load saved crops) ----------
def make_roi_dataset(
   data_path: str,
   roi_path: str,
   split: str = "train",
   batch_size: int = 16,
   cache: bool = True,
) -> tf.data.Dataset:
   """
   Build a tf.data pipeline yielding paired ROI crops and labels:
   ({"carpal": img, "metaph": img, "gender": int32}, age_months)

   Args:
      data_path (str): Root directory containing 'images/' and 'labels.csv'.
      roi_path (str): Base dir with saved crops: {roi_root}/{split}/{carpal,metaph}/*.png
      split (str): 'train' | 'validation' | 'test'
      batch_size (int): Batch size.
      cache (bool): Cache dataset in memory.

   Returns:
      tf.data.Dataset: Batches of paired crops and labels.
   """
   logger = get_logger(__name__)
   split = {"train": "train", "val": "validation", "validation":"validation", "test": "test"}[split.lower()]
   carpal_dir = os.path.join(roi_path, split, "carpal")
   metaph_dir = os.path.join(roi_path, split, "metaph")
   coords_csv = os.path.join(roi_path, split, "roi_coords.csv")
   
   # Expect original labels CSV from data_path
   labels_csv = os.path.join(data_path, f"{split}.csv")
   rows = read_csv_labels(labels_csv)
   
   # Build id -> path maps
   carpal_map = build_id_to_path(carpal_dir)
   metaph_map = build_id_to_path(metaph_dir)

   # Load ROI coordinate annotations when available
   coords_map = {}
   if os.path.exists(coords_csv):
      with open(coords_csv, newline="") as f:
         reader = csv.DictReader(f)
         for row in reader:
            image_id = row.get("image_id")
            if not image_id:
               continue
            try:
               carpal_box = (
                  int(row["carpal_y0"]), int(row["carpal_x0"]),
                  int(row["carpal_y1"]), int(row["carpal_x1"])
               )
               metaph_box = (
                  int(row["metaph_y0"]), int(row["metaph_x0"]),
                  int(row["metaph_y1"]), int(row["metaph_x1"])
               )
            except (ValueError, KeyError) as exc:
               logger.warning("Skipping malformed ROI coords for %s: %s", image_id, exc)
               continue
            coords_map[image_id] = (carpal_box, metaph_box)
   else:
      logger.warning("ROI coords file not found at %s", coords_csv)
   
   carpal_paths, metaph_paths, genders, ages, ids = [], [], [], [], []
   carpal_boxes, metaph_boxes = [], []
   missing_images = []
   missing_coords = []
   for r in rows:
      image_id = r["image_id"]
      carp = carpal_map.get(image_id, None)
      metp = metaph_map.get(image_id, None)
      
      if carp is None or metp is None:
         missing_images.append(image_id)
         continue

      coord_pair = coords_map.get(image_id)
      if coord_pair is None:
         missing_coords.append(image_id)
         coord_pair = ((-1, -1, -1, -1), (-1, -1, -1, -1))

      carpal_paths.append(carp)
      metaph_paths.append(metp)
      genders.append(r["male"])
      ages.append(r["age_months"])
      ids.append(image_id)
      carpal_boxes.append(coord_pair[0])
      metaph_boxes.append(coord_pair[1])
      
   if missing_images:
      logger.warning("%d ROI crops missing for %s split (examples: %s)", len(missing_images), split, missing_images[:5])
   if missing_coords:
      logger.warning("%d ROI coords missing for %s split (examples: %s)", len(missing_coords), split, missing_coords[:5])
   
   carpal_path_ds = tf.data.Dataset.from_tensor_slices(np.array(carpal_paths))   
   metaph_path_ds = tf.data.Dataset.from_tensor_slices(np.array(metaph_paths))
   gender_ds = tf.data.Dataset.from_tensor_slices(np.array(genders, dtype=np.int32))
   age_ds = tf.data.Dataset.from_tensor_slices(np.array(ages, dtype=np.float32))
   id_ds = tf.data.Dataset.from_tensor_slices(np.array(ids, dtype=np.str_))
   carpal_box_ds = tf.data.Dataset.from_tensor_slices(np.array(carpal_boxes, dtype=np.int32))
   metaph_box_ds = tf.data.Dataset.from_tensor_slices(np.array(metaph_boxes, dtype=np.int32))
   dataset = tf.data.Dataset.zip((carpal_path_ds, metaph_path_ds, gender_ds, age_ds, id_ds, carpal_box_ds, metaph_box_ds))
   options = tf.data.Options()
   options.experimental_deterministic = True
   dataset = dataset.with_options(options)
      
   def _load_pair(
      carpal_path: tf.Tensor,
      metaph_path: tf.Tensor,
      gender: tf.Tensor,
      age: tf.Tensor,
      img_id: tf.Tensor,
      carpal_box: tf.Tensor,
      metaph_box: tf.Tensor,
   ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
      """
      Loads and preprocesses a single image and its label.

      Args:
         carpal_path (tf.Tensor): tf.string file path to cropped carpal image.
         metaph_path (tf.Tensor): tf.string file path to cropped metaphalange image.
         gender (tf.Tensor): tf.int32 scalar {0,1}.
         age (tf.Tensor): tf.float32 scalar (months).
         img_id (tf.Tensor): tf.string scalar, numeric ID from CSV (e.g., "4516").
         carpal_box (tf.Tensor): tf.int32 tensor [4] with (y0,x0,y1,x1).
         metaph_box (tf.Tensor): tf.int32 tensor [4] with (y0,x0,y1,x1).

      Returns:
         (features, label):
            features = {
               "carpal":     tf.float32 [H,W,1],
               "metaph":     tf.float32 [H,W,1],
               "carpal_box": tf.int32   [4],
               "metaph_box": tf.int32   [4],
               "gender":     tf.int32   [],
               "image_id":   tf.string []
            }
            label = tf.float32 []  # age in months
            Note: ROI boxes are (-1,-1,-1,-1) when coordinates are unavailable.
      """
      c_img = zscore_norm(read_image_grayscale(carpal_path))  # [H,W,1], float32 in [0,1], z-score normalized
      m_img = zscore_norm(read_image_grayscale(metaph_path))
         
      features = {
         "carpal": c_img, 
         "metaph": m_img,
         "carpal_box": tf.cast(carpal_box, tf.int32),
         "metaph_box": tf.cast(metaph_box, tf.int32),
         "gender": tf.cast(gender, tf.int32),
         "image_id": img_id
      }
      return features, tf.cast(age, tf.float32)
   
   dataset = dataset.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
   if cache:
      dataset = dataset.cache()
   dataset = dataset.batch(batch_size, drop_remainder=False)
   dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
   return dataset
