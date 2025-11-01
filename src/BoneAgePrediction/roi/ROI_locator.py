#TODO: add docstring explanation for this module. write in details and explain each function


from typing import Dict
import os, csv
import tensorflow as tf
import keras

from BoneAgePrediction.utils.dataset_loader import make_dataset
from BoneAgePrediction.utils.logger import get_logger, mirror_keras_stdout_to_file

from BoneAgePrediction.visualization.gradcam import compute_GradCAM
from BoneAgePrediction.visualization.overlay import overlay_cam_on_image

from BoneAgePrediction.roi.ROI_extract import extract_rois_from_heatmap

logger = get_logger(__name__)

CHECKPOINTS_BASE_DIR = "experiments/checkpoints"

def train_locator_and_save_rois(
   config: Dict,
   split: str,
   out_root: str,
) -> None:
   """
   Load a pretrained GlobalCNN, compute Grad-CAM per image to extract
   two ROI crops, and save them.

   Args:
      config (Dict): Loaded config (roi_only.yaml).
      split (str): 'train' | 'validation' | 'test'.
      out_root (str): Base folder to save crops (e.g., data/processed/cropped_rois).

   Produces:
      {out_root}/{split}/carpal/{image_id}.png
      {out_root}/{split}/metaph/{image_id}.png
      {out_root}/{split}/roi_coords.csv    # (image_id, y0,x0,y1,x1 for both ROIs)
   """
   
   mirror_keras_stdout_to_file()
   
   # -----------------------
   # Dataset for locator inference
   # -----------------------
   data_cfg = config.data
   roi_cfg = config.roi
   locator_cfg = roi_cfg.locator
   extractor_cfg = roi_cfg.extractor
   
   data_path = data_cfg.data_path
   image_size = data_cfg.image_size
   keep_aspect_ratio = data_cfg.keep_aspect_ratio
   batch_size = data_cfg.batch_size
   clahe = data_cfg.clahe
   augment = data_cfg.augment
   
   ds = make_dataset(
      data_path = data_path, 
      split = split,
      image_size = image_size,
      keep_aspect_ratio = keep_aspect_ratio, 
      batch_size = batch_size,
      clahe = clahe,
      augment = augment if split == "train" else False,
   )
   logger.info("Created %s dataset for ROI locator from %s.", split, data_path) 

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
      raise FileNotFoundError(
         f"Pretrained ROI locator weights not found at {pretrained_model_path} (from config value '{raw_model_path}')."
      )

   roi_loc_model = keras.models.load_model(
      pretrained_model_path,
      compile=False,
   )
   logger.info(
      "Loaded pretrained GlobalCNN model for ROI locator from %s (config='%s', split=%s).",
      pretrained_model_path,
      raw_model_path,
      split,
   )


   # --- Prepare output dirs
   carpal_dir = os.path.join(out_root, split, "carpal")
   metaph_dir = os.path.join(out_root, split, "metaph")
   os.makedirs(carpal_dir, exist_ok=True)
   os.makedirs(metaph_dir, exist_ok=True)
   
   save_heatmaps = extractor_cfg.save_heatmaps
   if save_heatmaps:
      os.makedirs(os.path.join(out_root, split, "heatmaps"), exist_ok=True)

   # --- Iterate once over split, save crops
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
         image_viz = features.get("image_viz", image)     # [H,W,1], float32 in [0,1]
         image_id = features["image_id"]                  # [ ], string
         
         # Compute Grad-CAM on the locator
         cam = compute_GradCAM(
            model=roi_loc_model,
            image=image,
            target_layer_name=None,
            target_index=0,
         )  # [H,W] in [0,1]
         
         # Extract ROIs from the heatmap
         roi_size=extractor_cfg.roi_size
         heatmap_threshold=extractor_cfg.heatmap_threshold
         rois = extract_rois_from_heatmap(
               heatmap=cam,
               image=image_viz,
               roi_size=roi_size,
               carpal_margin=0.45, # extra border around peak box (fraction of shorter side)
               meta_mask_radius=0.25, # mask radius (fraction of shorter side) to hide carpal when finding metacarpal
               heatmap_threshold=heatmap_threshold,
         ) # Dict with "carpal" and "metaph" entries
         
         # --- Use the true image_id from CSV (e.g., "4516")
         img_id = image_id.numpy().decode("utf-8")
         
         # Save crops as {image_id}.png
         _save_png(os.path.join(carpal_dir, f"{img_id}.png"), rois["carpal"]["crop"])
         _save_png(os.path.join(metaph_dir, f"{img_id}.png"), rois["metaph"]["crop"])

         # Optionally save heatmap overlay
         if save_heatmaps:
            overlay_rgb = overlay_cam_on_image(
               gray_img=image_viz,
               cam=cam,
            )
            _save_png(
               os.path.join(out_root, split, "heatmaps", f"{img_id}.png"),
               overlay_rgb,
            )

         (y0,x0,y1,x1)   = rois["carpal"]["box"]
         (y0b,x0b,y1b,x1b) = rois["metaph"]["box"]
         w.writerow([img_id, y0,x0,y1,x1, y0b,x0b,y1b,x1b])

def _save_png(path: str, img: tf.Tensor) -> None:
   """
   Save an image tensor or array to PNG. Float inputs are clipped to [0,1]
   before conversion to uint8.

   Args:
      path (str): Destination file path.
      img (tf.Tensor): Image tensor or array (float or uint8).
   """
   x = tf.convert_to_tensor(img)
   if x.dtype != tf.uint8:
      if x.dtype.is_floating:
         x = tf.clip_by_value(x, 0.0, 1.0)
         x = tf.image.convert_image_dtype(x, dtype=tf.uint8)
      else:
         x = tf.cast(x, tf.uint8)
   tf.io.write_file(path, tf.io.encode_png(x))
