"""Utilities for overlaying class-activation heatmaps on grayscale images."""

import numpy as np
import tensorflow as tf
import matplotlib.cm as cm

def overlay_cam_on_image(
   gray_img: tf.Tensor | np.ndarray,
   cam: tf.Tensor | np.ndarray,
   alpha: float = 0.35,
   gamma: float = 1.0,
   cmap_name: str = "jet",
) -> np.ndarray:
   """
   Blend a CAM heatmap onto a grayscale or RGB image.

   Args:
      gray_img: Image tensor/array shaped `[H, W]`, `[H, W, 1]`, or `[H, W, 3]`.
      cam: Rank-2 heatmap with the same spatial resolution as `gray_img`
         (or will be resized to it).
      alpha: Heatmap opacity multiplier.
      gamma: Optional gamma correction applied to the base image.
      cmap_name: Matplotlib colormap name used for the heatmap.

   Returns:
      `np.ndarray` uint8 RGB image with the CAM blended in.

   Raises:
      ValueError: If the input shapes do not match the expected ranks.
   """

   gray = tf.convert_to_tensor(gray_img)
   cam = tf.convert_to_tensor(cam, dtype=tf.float32)

   if gray.dtype.is_integer:
      gray = tf.image.convert_image_dtype(gray, tf.float32)
   else:
      gray = tf.cast(gray, tf.float32)

   if gray.shape.rank == 2:
      gray = gray[..., tf.newaxis]
   if gray.shape.rank != 3 or gray.shape[-1] not in (1, 3):
      raise ValueError("gray_img must be [H,W] or [H,W,{1|3}].")

   heatmap = tf.squeeze(cam)
   if heatmap.shape.rank != 2:
      raise ValueError("cam must be rank-2 heatmap [H,W].")

   target_hw = tf.shape(gray)[0:2]
   heatmap = tf.image.resize(
      heatmap[..., tf.newaxis],
      size=target_hw,
      method="bilinear",
   )
   heatmap = tf.squeeze(heatmap, axis=-1)
   heatmap = tf.clip_by_value(heatmap, 0.0, 1.0)

   colormap = cm.get_cmap(cmap_name)
   colormap_lut = tf.constant(colormap(np.arange(256))[:, :3], dtype=tf.float32)
   heatmap_indices = tf.cast(tf.round(heatmap * 255.0), tf.int32)
   heatmap_rgb = tf.gather(colormap_lut, heatmap_indices)  # [H,W,3]

   base = gray if gray.shape[-1] == 3 else tf.repeat(gray, repeats=3, axis=-1)
   if gamma != 1.0:
      base = tf.pow(tf.clip_by_value(base, 0.0, 1.0), gamma)

   overlay = heatmap_rgb * float(alpha) + base
   overlay = tf.clip_by_value(overlay, 0.0, 1.0)
   return (overlay.numpy() * 255.0).astype(np.uint8)
