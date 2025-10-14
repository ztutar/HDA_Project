from typing import Tuple, Dict
import numpy as np
import tensorflow as tf

def peak_loc_heatmap(heatmap: tf.Tensor) -> Tuple[int, int]:
   """
   Find (y,x) location of maximum in a CAM heatmap.

   Args:
      heatmap (tf.Tensor): [H,W] float32 heatmap in [0,1].

   Returns:
      Tuple[int,int]: (y,x) indices of the peak.
   """
   idx = tf.argmax(tf.reshape(heatmap, [-1]))
   h, w = heatmap.shape
   y = idx // w
   x = idx % w
   return int(y), int(x)

def square_box_around(center_y: int, center_x: int,
                     size: int, margin_frac: float,
                     H: int, W: int) -> Tuple[int, int, int, int]:
   """
   Build a square crop around a center with a fractional margin.

   Args:
      center_y, center_x (int): Peak coordinates.
      size (int): Target crop size (square).
      margin_frac (float): Extra context as a fraction of min(H,W).
      H, W (int): Heatmap/image dimensions.

   Returns:
      (y0, x0, y1, x1): Integer coordinates, clipped to image bounds.
   """
   r = int(min(H,W) * margin_frac / 2.0)
   y0 = max(0, center_y - r)
   x0 = max(0, center_x - r)
   y1 = min(H, center_y + r)
   x1 = min(W, center_x + r)
   
   # Ensure non-degenerate box
   if y1 - y0 < 3:
      y1 = min(H, y0 + 3)
   if x1 - x0 < 3:
      x1 = min(W, x0 + 3)
   return y0, x0, y1, x1

def mask_disk(heatmap: tf.Tensor, cy: int, cx: int,
            radius_frac: float) -> tf.Tensor:
   """
   Mask a disk region around (cy,cx) to suppress the first ROI when searching the second.

   Args:
      heatmap (tf.Tensor): [H,W] CAM heatmap.
      cy, cx (int): Center to mask.
      radius_frac (float): Disk radius as a fraction of min(H,W).

   Returns:
      tf.Tensor: New heatmap with the disk zeroed out.
   """
   H, W = heatmap.shape
   yy, xx = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
   r = tf.cast(radius_frac * tf.cast(tf.minimum(H,W), tf.float32), tf.int32)
   dist = tf.sqrt(tf.square(tf.cast(yy - cy, tf.float32)) + tf.square(tf.cast(xx - cx, tf.float32)))
   mask = tf.cast(dist > r, tf.float32)
   return heatmap * mask

def extract_rois_from_heatmap(
   heatmap: tf.Tensor,
   image: tf.Tensor,
   roi_size: int,
   carpal_margin: float,
   meta_mask_radius: float,
   heatmap_threshold: float,
) -> Dict[str, tf.Tensor]:
   """
   Given a Grad-CAM heatmap on the full image, extract two square ROI crops:
   1) Carpal (first/primary peak)
   2) Metacarpal/phalange (second peak after masking the first region)

   Args:
      heatmap (tf.Tensor): [H,W] Grad-CAM in [0,1].
      img (tf.Tensor): [H,W,1] full image tensor (float32).
      roi_size (int): Output crop size (square).
      carpal_margin (float): Context margin for ROI-1 box (fraction of min(H,W)).
      meta_mask_radius (float): Mask radius to suppress ROI-1 when finding ROI-2 (fraction).
      heatmap_threshold (float): Reject peaks when cam value < threshold.

   Returns:
      Dict[str,Dict]: {
         "carpal": {"crop": tf.Tensor[roi_size,roi_size,1], "box": (y0,x0,y1,x1)},
         "metaph": {"crop": tf.Tensor[roi_size,roi_size,1], "box": (y0,x0,y1,x1)}
      }
   """
   H, W = heatmap.shape
   # --- 1) Carpal from top peak ---
   cy1, cx1 = peak_loc_heatmap(heatmap)
   if heatmap[cy1, cx1] < heatmap_threshold:
      # fallback: center
      cy1, cx1 = H // 2, W // 2
   y0_a, x0_a, y1_a, x1_a = square_box_around(cy1, cx1, roi_size, carpal_margin, H, W)
   crop_a = tf.image.resize(image[y0_a:y1_a, x0_a:x1_a, :], size=(roi_size, roi_size), antialias=True)
   
   # --- 2) Metacarpal/phalange from second peak ---
   heatmap2 = mask_disk(heatmap, cy1, cx1, radius_frac=meta_mask_radius)
   cy2, cx2 = peak_loc_heatmap(heatmap2)
   if heatmap2[cy2, cx2] < heatmap_threshold:
      # fallback: opposite quadrant
      cy2, cx2 = max(0, H - cy1), max(0, W - cx1)
   y0_b, x0_b, y1_b, x1_b = square_box_around(cy2, cx2, roi_size, carpal_margin, H, W)
   crop_b = tf.image.resize(image[y0_b:y1_b, x0_b:x1_b, :], size=(roi_size, roi_size), antialias=True)
   
   rois = {
      "carpal": {"crop": tf.cast(crop_a, tf.float32), "box": (int(y0_a), int(x0_a), int(y1_a), int(x1_a))},
      "metaph": {"crop": tf.cast(crop_b, tf.float32), "box": (int(y0_b), int(x0_b), int(y1_b), int(x1_b))}
   }
   return rois