# ROI crop generator (runs once to save crops)

from typing import Tuple, Dict, List
import os, csv
import tensorflow as tf
import numpy as np
from keras import optimizers

from BoneAgePrediction.data.dataset_loader import make_dataset
from BoneAgePrediction.models.B0_global_cnn import build_GlobalCNN
from BoneAgePrediction.training.losses import get_loss
from BoneAgePrediction.training.metrics import mae, rmse
from BoneAgePrediction.visualization.gradcam import compute_GradCAM
from BoneAgePrediction.roi.roi_extract import extract_rois_from_heatmap

def train_locator_and_save_rois(
   config: Dict,
   split: str,
   out_root: str,
   save_heatmaps: bool = False,
) -> None:
   """
   Train a locator CNN on full images for regression, then compute 
   Grad-CAM per image to extract two ROI crops and save them.

   Args:
      config (Dict): Loaded config (roi_only.yaml).
      split (str): 'train' | 'validation' | 'test'.
      out_root (str): Base folder to save crops (e.g., data/processed/cropped_rois).
      save_heatmaps (bool): If True, also save heatmap PNGs for inspection.

   Produces:
      {out_root}/{split}/carpal/{image_id}.png
      {out_root}/{split}/metaph/{image_id}.png
      {out_root}/{split}/roi_coords.csv    # (image_id, y0,x0,y1,x1 for both ROIs)
   """
   # --- 1) Datasets for ROI locator
   data_cfg = config.data
   roi_cfg = config.roi
   locator_cfg = roi_cfg.locator
   extractor_cfg = roi_cfg.extractor
   
   data_path = data_cfg.data_path
   target_h = data_cfg.target_h
   target_w = data_cfg.target_w
   keep_aspect_ratio = data_cfg.keep_aspect_ratio
   pad_value = data_cfg.pad_value
   batch_size = data_cfg.batch_size
   clahe = data_cfg.clahe
   augment = data_cfg.augment
   cache = data_cfg.cache
   
   ds = make_dataset(
      data_path = data_path, 
      split = split,
      target_h = target_h, 
      target_w = target_w,
      keep_aspect_ratio = keep_aspect_ratio, 
      pad_value = pad_value,
      batch_size = batch_size, 
      clahe = clahe,
      augment = augment if split == "train" else False,
      cache = cache
   )

   # --- 2) ROI Locator model (CNN)
   input_shape = (target_h, target_w, 1)
   
   roi_loc_model = build_GlobalCNN(
      input_shape = input_shape,
      channels = tuple(locator_cfg.channels),
      dense_units = locator_cfg.dense_units,
      name="ROI_Locator_CNN",
   )
   optimizer = optimizers.Adam(learning_rate=locator_cfg.learning_rate)
   roi_loc_model.compile(optimizer=optimizer, loss=get_loss("huber", 10.0), metrics=[mae(), rmse()])

   if split == "train":
      # quick fit to give CAM a sensible signal
      roi_loc_model.fit(ds, epochs=locator_cfg.epochs, verbose=1)
   else:
      # For val/test, just do 1 pass to initialize weights structure (optional).
      _ = roi_loc_model.predict(ds.take(1), verbose=0)

   # --- 3) Prepare output dirs
   carpal_dir = os.path.join(out_root, split, "carpal")
   metaph_dir = os.path.join(out_root, split, "metaph")
   os.makedirs(carpal_dir, exist_ok=True)
   os.makedirs(metaph_dir, exist_ok=True)
   if save_heatmaps:
      os.makedirs(os.path.join(out_root, split, "heatmaps"), exist_ok=True)

   # --- 4) Iterate once over split, save crops
   coords_path = os.path.join(out_root, split, "roi_coords.csv")
   with open(coords_path, "w", newline="") as f:
      w = csv.writer(f)
      w.writerow([
         "image_id",
         "carpal_y0","carpal_x0","carpal_y1","carpal_x1",
         "metaph_y0","metaph_x0","metaph_y1","metaph_x1",
      ])
      for features, _ in ds.unbatch():
         image = features["image"]                        # [H,W,1], float32
         image_id = features["image_id"]                  # [ ], string
         
         # Compute Grad-CAM on the locator
         cam = compute_GradCAM(roi_loc_model, image, target_layer_name=None)  # [H,W] in [0,1]

         rois = extract_rois_from_heatmap(
               heatmap=cam,
               image=image,
               roi_size=extractor_cfg.roi_size,
               carpal_margin=extractor_cfg.carpal_margin,
               meta_mask_radius=extractor_cfg.meta_mask_radius,
               heatmap_threshold=extractor_cfg.heatmap_threshold,
         )
         # --- Use the true image_id from CSV (e.g., "4516")
         img_id = image_id.numpy().decode("utf-8")
         
         # Save crops as {image_id}.png
         save_png(os.path.join(carpal_dir, f"{img_id}.png"), rois["carpal"]["crop"])
         save_png(os.path.join(metaph_dir, f"{img_id}.png"), rois["metaph"]["crop"])

         # Optionally save heatmap
         if save_heatmaps:
               save_png(os.path.join(out_root, split, "heatmaps", f"{img_id}.png"), cam[..., None])

         (y0,x0,y1,x1)   = rois["carpal"]["box"]
         (y0b,x0b,y1b,x1b) = rois["metaph"]["box"]
         w.writerow([img_id, y0,x0,y1,x1, y0b,x0b,y1b,x1b])

def save_png(path: str, img01: tf.Tensor) -> None:
   """
   Save a single-channel float32 image [H,W,1] in [0,1] to PNG.

   Args:
      path (str): Destination file path.
      img01 (tf.Tensor): Image in [0,1].
   """
   x = tf.clip_by_value(img01, 0.0, 1.0)
   x = tf.image.convert_image_dtype(x, dtype=tf.uint8)
   tf.io.write_file(path, tf.io.encode_png(x))
