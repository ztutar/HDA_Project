"""Utilities for running the ROI locator network and persisting its predictions.

The locator takes preprocessed hand radiographs, generates Grad-CAM heatmaps to
highlight salient regions, then extracts carpal and metacarpal crops that will
feed downstream pipelines. This module hides the mechanics of loading a
pretrained locator checkpoint, iterating over TF datasets, computing Grad-CAM,
extracting crops, and saving the resulting PNG files (optionally with heatmap
overlays).
"""


from typing import Dict
import os
import tensorflow as tf
import keras
from pathlib import Path
import pandas as pd

from BAP.utils.dataset_loader import make_dataset
from BAP.utils.config import ProjectConfig

#from BAP.utils.logger import get_logger, mirror_keras_stdout_to_file

from BAP.visualization.gradcam import compute_GradCAM
from BAP.visualization.overlay import overlay_cam_on_image

from BAP.roi.ROI_extract import extract_rois_from_heatmap

#logger = get_logger(__name__)

CHECKPOINTS_BASE_DIR = "experiments/checkpoints"

def train_locator_and_save_rois(
   data_path: Path,
   roi_paths: Dict,
   config: ProjectConfig,
   split: str,
) -> None:
   """Run the pretrained ROI locator on a dataset split and save crops as PNGs.

   The function builds a TF dataset for the requested `split`, loads the locator
   checkpoint indicated in the configuration, and performs a single inference
   pass. For every image it computes Grad-CAM, extracts carpal and metacarpal
   ROIs from the heatmap, and writes them to the directories provided in
   `roi_paths`. When heatmap saving is enabled, the colorized overlay is also
   written to disk, mirroring the image IDs from the metadata CSV.

   Args:
      data_path: Directory containing the raw images for the given split.
      roi_paths: Mapping with `"carpal"`, `"metaph"`, and optionally `"heatmaps"`
         destinations where PNGs will be saved.
      config: Full project configuration providing dataset, locator, and
         extractor settings.
      split: Dataset split name (e.g., `"train"`, `"val"`, `"test"`) used to
         select the CSV metadata file and augmentations.

   Raises:
      ValueError: If the pretrained model path is missing in the configuration.
      FileNotFoundError: When the specified checkpoint cannot be located on disk.
   """
   
   #mirror_keras_stdout_to_file()
   
   # -----------------------
   # Dataset for locator inference
   # -----------------------
   data_cfg = config.data
   roi_cfg = config.roi
   locator_cfg = roi_cfg.locator
   extractor_cfg = roi_cfg.extractor
   
   image_size = data_cfg.image_size
   clahe = data_cfg.clahe
   augment = data_cfg.augment
   
   ds = make_dataset(
      image_dir=data_path,
      metadata=pd.read_csv(f"data/metadata/{split}.csv"),
      image_size=image_size,
      clahe=clahe,
      augment=augment if split == "train" else False,
   )
   #logger.info("Created %s dataset for ROI locator from %s.", split, data_path) 

   # -----------------------
   # Model for ROI locator
   # -----------------------
   raw_model_path = locator_cfg.pretrained_model_path
   if not raw_model_path:
      raise ValueError(
         "`roi.locator.pretrained_model_path` is empty; provide a relative path under "
         f"{CHECKPOINTS_BASE_DIR} or an absolute path."
      )

   pretrained_model_path = raw_model_path
   if not os.path.isabs(pretrained_model_path):
      pretrained_model_path = os.path.join(CHECKPOINTS_BASE_DIR, pretrained_model_path)

   pretrained_model_path = os.path.normpath(pretrained_model_path)
   if not os.path.exists(pretrained_model_path):
      raise FileNotFoundError(f"Pretrained ROI locator weights not found at {pretrained_model_path} (from config value '{raw_model_path}').")

   roi_loc_model = keras.models.load_model(pretrained_model_path, compile=False)
   #logger.info(
   #   "Loaded pretrained GlobalCNN model for ROI locator from %s (config='%s', split=%s).",
   #   pretrained_model_path,
   #   raw_model_path,
   #   split,
   #)

   # --- Prepare output dirs
   carpal_dir = roi_paths["carpal"]
   metaph_dir = roi_paths["metaph"]
   os.makedirs(carpal_dir, exist_ok=True)
   os.makedirs(metaph_dir, exist_ok=True)
   
   save_heatmaps = extractor_cfg.save_heatmaps
   if save_heatmaps:
      heatmap_dir = roi_paths["heatmaps"]
      os.makedirs(heatmap_dir, exist_ok=True)

   # --- Iterate once over split, save crops
   for features, _ in ds:
      image = features["image"]                        # [H,W,1] 
      image_viz =features.get("image_viz", image)      # [H,W,1], float32 in [0,1]
      image_id = features["image_id"]                  
      
      # Compute Grad-CAM on the locator
      cam = compute_GradCAM(model=roi_loc_model, image=image)  # [H,W]
      
      # Extract ROIs from the heatmap
      roi_size=extractor_cfg.roi_size
      heatmap_threshold=extractor_cfg.heatmap_threshold
      rois = extract_rois_from_heatmap(
            heatmap=cam,
            image=image_viz,
            roi_size=roi_size,
            carpal_margin=0.48, # extra border around peak box (fraction of shorter side)
            meta_mask_radius=0.35, # mask radius (fraction of shorter side) to hide carpal when finding metacarpal
            heatmap_threshold=heatmap_threshold,
      ) # Dict with "carpal" and "metaph" entries
      
      # --- Use the true image_id from CSV (e.g., "4516")
      img_id = image_id.numpy().decode("utf-8")
      
      # Save crops as {image_id}.png
      _save_png(os.path.join(carpal_dir, f"{img_id}.png"), rois["carpal"])
      _save_png(os.path.join(metaph_dir, f"{img_id}.png"), rois["metaph"])

      # Optionally save heatmap overlay
      if save_heatmaps:
         overlay_rgb = overlay_cam_on_image(gray_img=image, cam=cam)
         _save_png(os.path.join(heatmap_dir, f"{img_id}.png"), overlay_rgb)



def _save_png(path: str, img: tf.Tensor) -> None:
   """Normalize a tensor to uint8 if needed and write it as a PNG file."""
   x = tf.convert_to_tensor(img)
   if x.dtype != tf.uint8:
      if x.dtype.is_floating:
         x = tf.clip_by_value(x, 0.0, 1.0)
         x = tf.image.convert_image_dtype(x, dtype=tf.uint8)
      else:
         x = tf.cast(x, tf.uint8)
   tf.io.write_file(path, tf.io.encode_png(x))
