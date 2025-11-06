#TODO: add docstring explanation for this module. write in details and explain each function


from typing import Tuple, Dict, List
import tensorflow as tf


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
      meta_mask_radius (float): Width fraction for the bottom-anchored mask used when finding ROI-2.
      heatmap_threshold (float): Reject peaks when cam value < threshold.

   Returns:
      Dict[str,Dict]: {
         "carpal": {"crop": tf.Tensor[roi_size,roi_size,1], "box": (y0,x0,y1,x1)},
         "metaph": {"crop": tf.Tensor[roi_size,roi_size,1], "box": (y0,x0,y1,x1)}
      }
   """
   heatmap = tf.convert_to_tensor(heatmap, dtype=tf.float32)
   image = tf.convert_to_tensor(image, dtype=tf.float32)

   H, W = heatmap.shape
   # --- Determine primary peaks and ensure carpal is the lowest peak ---
   peak_candidates = _top_k_peak_locs(heatmap, k=8)
   primary_cy, _ = peak_candidates[0]
   carpal_cy, carpal_cx = max(peak_candidates[:2], key=lambda loc: loc[0])
   if carpal_cy <= primary_cy:
      # First two peaks are not below the primary one—scan additional peaks.
      for cy, cx in peak_candidates[2:]:
         if cy > carpal_cy:
            carpal_cy, carpal_cx = cy, cx
            break
   if carpal_cy < H // 2:
      lower_peaks = [loc for loc in peak_candidates if loc[0] >= H // 2]
      if lower_peaks:
         carpal_cy, carpal_cx = max(lower_peaks, key=lambda loc: loc[0])
   
   if heatmap[carpal_cy, carpal_cx] < heatmap_threshold:
      # fallback: center of the image
      carpal_cy, carpal_cx = H // 2, W // 2
   y0_a, x0_a, y1_a, x1_a = _square_box_around(carpal_cy, carpal_cx, carpal_margin, H, W)
   crop_a = tf.image.resize(
      image[y0_a:y1_a, x0_a:x1_a, :],
      size=(roi_size, roi_size),
      antialias=True,
   )
   
   # --- 2) Metacarpal/phalange from second peak after rectangular masking ---
   heatmap2 = _mask_rect_to_bottom(
      heatmap,
      carpal_cy,
      carpal_cx,
      radius_frac=meta_mask_radius,
   )
   cy2, cx2 = _peak_loc_heatmap(heatmap2)
   if heatmap2[cy2, cx2] < heatmap_threshold:
      # fallback: opposite quadrant relative to carpal
      cy2, cx2 = max(0, H - carpal_cy), max(0, W - carpal_cx)
   y0_b, x0_b, y1_b, x1_b = _square_box_around(cy2, cx2, carpal_margin, H, W)
   crop_b = tf.image.resize(image[y0_b:y1_b, x0_b:x1_b, :], size=(roi_size, roi_size), antialias=True)
   
   rois = {
      "carpal": {"crop": tf.cast(crop_a, tf.float32), "box": (int(y0_a), int(x0_a), int(y1_a), int(x1_a))},
      "metaph": {"crop": tf.cast(crop_b, tf.float32), "box": (int(y0_b), int(x0_b), int(y1_b), int(x1_b))}
   }
   return rois


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def _peak_loc_heatmap(heatmap: tf.Tensor) -> Tuple[int, int]:
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

def _top_k_peak_locs(heatmap: tf.Tensor, k: int) -> List[Tuple[int, int]]:
   """
   Return the (y,x) locations of the top-k peaks in the heatmap.

   Args:
      heatmap (tf.Tensor): [H,W] float32 heatmap.
      k (int): Number of peaks requested.

   Returns:
      List[Tuple[int,int]]: Peak locations sorted by descending activation.
   """
   h, w = heatmap.shape
   if h is None or w is None:
      raise ValueError("Heatmap spatial dimensions must be statically known.")
   flat = tf.reshape(heatmap, [-1])
   flat_size = heatmap.shape.num_elements()
   if flat_size is None:
      raise ValueError("Heatmap spatial dimensions must be statically known.")
   flat_size = int(flat_size)
   k = max(1, min(int(k), flat_size))
   _, indices = tf.math.top_k(flat, k=k, sorted=True)
   peaks: List[Tuple[int, int]] = []
   for i in range(k):
      idx = int(indices[i])
      y = idx // w
      x = idx % w
      peaks.append((int(y), int(x)))
   return peaks

def _square_box_around(
   center_y: int, 
   center_x: int,
   margin_frac: float, 
   H: int, 
   W: int,
) -> Tuple[int, int, int, int]:
   """
   Build a square crop around (center_y, center_x) using a margin fraction of image size.

   Args:
      center_y, center_x (int): Peak coordinates.
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

def _mask_rect_to_bottom(
   heatmap: tf.Tensor,
   cy: int,
   cx: int,
   radius_frac: float,
) -> tf.Tensor:
   """
   Mask a bottom-anchored rectangle around (cy,cx) to suppress the carpal ROI when searching the second.

   Args:
      heatmap (tf.Tensor): [H,W] CAM heatmap.
      cy, cx (int): Center to mask.
      radius_frac (float): Rectangle width as a fraction of min(H,W).

   Returns:
      tf.Tensor: New heatmap with the rectangle zeroed out.
   """
   if radius_frac <= 0:
      return heatmap
   H, W = heatmap.shape
   if H == 0 or W == 0:
      return heatmap
   min_dim = float(min(H, W))
   rect_width = radius_frac * min_dim
   half_width = rect_width / 2.0
   if half_width <= 0:
      return heatmap
   
   yy, xx = tf.meshgrid(tf.range(H), tf.range(W), indexing="ij")
   yy = tf.cast(yy, tf.float32)
   xx = tf.cast(xx, tf.float32)
   center_y = tf.cast(cy, tf.float32)
   center_x = tf.cast(cx, tf.float32)
   top = tf.maximum(0.0, center_y - half_width)
   left = center_x - rect_width
   right = center_x + rect_width
   
   within_x = tf.logical_and(xx >= left, xx <= right)
   within_y = yy >= top
   rect_mask = tf.logical_and(within_x, within_y)
   keep_mask = tf.logical_not(rect_mask)
   return heatmap * tf.cast(keep_mask, heatmap.dtype)
