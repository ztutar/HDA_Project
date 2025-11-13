"""ROI extraction helpers for hand bone age pipelines.

The routines in this module take a single-channel heatmap produced by a
localization network and the corresponding image and derive two square 
regions of interest (ROIs): a carpal ROI that is centered on the most prominent 
activation peak in the lower half of the heatmap, and a metacarpal ROI that is 
searched for after masking out a rectangular area beneath the first peak. The 
utilities handle peak selection, bounding-box construction, and masking.
"""

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
   """Crop carpal and metacarpal ROIs based on heatmap peak locations.

   The procedure first converts the supplied heatmap and image to float tensors,
   then locates the highest peak to estimate the general carpal area. The second
   ROI is discovered by masking a rectangular region extending downward from
   that first peak and finding the next strongest activation, which tends to
   correspond to the metacarpal/phalange area.  Both ROIs are cropped with a
   square whose side length is proportional to the image size and resized to a
   fixed `roi_size` for downstream models.

   Args:
      heatmap: 2D tensor ``[H, W]`` containing localization scores.
      image: Tensor ``[H, W, C]`` holding the source image that aligns with the
         heatmap.
      roi_size: Target side length (in pixels) of the resized ROI crops.
      carpal_margin: Fraction of the minimum spatial dimension used to derive
         the square crop radius around the detected carpal center.
      meta_mask_radius: Fractional width of the rectangular mask applied before
         searching for the metacarpal peak.  Higher values hide a wider band.
      heatmap_threshold: Minimum acceptable activation for selecting a peak;
         if the activation falls below this value, the function falls back to
         the spatial center of the image to avoid empty crops.

   Returns:
      A dictionary with two float32 tensors of shape ``[roi_size, roi_size, C]``:
      ``"carpal"`` for the wrist region and ``"metaph"`` for the
      metacarpal/phalange region.
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
      "carpal": tf.cast(crop_a, tf.float32),
      "metaph": tf.cast(crop_b, tf.float32)
   }
   return rois


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def _peak_loc_heatmap(heatmap: tf.Tensor) -> Tuple[int, int]:
   """Return the coordinates of the global maximum in the heatmap.

   The tensor is flattened to find the argmax index, which is then converted
   back to ``(row, column)`` coordinates.  The function assumes static spatial
   dimensions so it can operate inside TensorFlow graphs without shape tracing.

   Args:
      heatmap: 2D tensor containing scalar activation values.

   Returns:
      Tuple ``(y, x)`` with integer indices of the maximum activation.
   """


   idx = tf.argmax(tf.reshape(heatmap, [-1]))
   h, w = heatmap.shape
   y = idx // w
   x = idx % w
   return int(y), int(x)

def _top_k_peak_locs(heatmap: tf.Tensor, k: int) -> List[Tuple[int, int]]:
   """Return the ``k`` highest-scoring peak coordinates from the heatmap.

   It validates that the heatmap shape is statically known, flattens it, and
   uses ``tf.math.top_k`` to obtain the indices for the strongest activations.
   Each flat index is translated back to ``(row, column)`` coordinates and
   collected in descending order of activation.

   Args:
      heatmap: 2D tensor with localization activations.
      k: Desired number of peaks; automatically clamped to ``[1, H*W]``.

   Returns:
      List of length ``k`` containing ``(y, x)`` tuples ordered by peak score.
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
   """Compute a clipped square bounding box around a center point.

   The side length is derived from ``margin_frac * min(H, W)`` so the box scales
   with the image resolution.  Bounds are clipped to the image size and enforced
   to be at least three pixels wide/high to avoid degenerate crops that break
   subsequent resize operations.

   Args:
      center_y: Vertical coordinate of the box center.
      center_x: Horizontal coordinate of the box center.
      margin_frac: Fraction of the smaller image dimension used to expand the
         square in all directions.
      H: Image height.
      W: Image width.

   Returns:
      Tuple ``(y0, x0, y1, x1)`` describing the inclusive-exclusive crop bounds.
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
   """Zero out a vertical rectangular band beneath a reference peak.

   The rectangle spans horizontally ``2 * radius_frac * min(H, W)`` pixels and
   extends from ``cy - width/2`` down to the bottom of the heatmap.  This helps
   suppress the already-discovered carpal region so that subsequent peaks are
   forced to appear elsewhere when searching for the metacarpal ROI.

   Args:
      heatmap: Source heatmap tensor that will be masked.
      cy: Row index of the reference peak.
      cx: Column index of the reference peak.
      radius_frac: Relative half-width controlling how wide the masked band is;
         non-positive values leave the heatmap unchanged.

   Returns:
      A tensor with the same shape as ``heatmap`` where the masked region is set
      to zero (or the tensor's zero-equivalent value).
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
