#TODO: add docstring explanation for this module. write in details and explain each function


from typing import Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def overlay_cam_on_image(
   gray_img: tf.Tensor | np.ndarray,
   cam: tf.Tensor | np.ndarray,
   alpha: float = 0.35,
   gamma: float = 1.0,
   cmap_name: str = "jet",
) -> np.ndarray:
   """
   Create an RGB overlay using a matplotlib colormap (e.g., 'jet', 'magma').

   Args:
      gray_img (tf.Tensor | np.ndarray): [H,W] or [H,W,1], float32/float64.
      cam (tf.Tensor | np.ndarray): [H,W], Grad-CAM heatmap.
      alpha (float): Heatmap opacity factor (0..1).
      gamma (float): Gamma correction for the grayscale base.
      cmap_name (str): Matplotlib colormap name.

   Returns:
      np.ndarray: [H,W,3] uint8 RGB image ready to save/show.
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

def show_cam_triptych(
   gray_img: tf.Tensor | np.ndarray,
   cam: tf.Tensor | np.ndarray,
   overlay_rgb: np.ndarray,
   titles: Tuple[str, str, str] = ("X-ray", "Grad-CAM", "Overlay"),
   figsize: Tuple[int, int] = (12, 4),
   cmap_name: str = "gray",
) -> None:
   """
   Display a 3-panel figure: original, heatmap, and overlay.

   Args:
      gray_img: [H,W] or [H,W,1] image.
      cam: [H,W] Grad-CAM map.
      overlay_rgb: [H,W,3] uint8 overlay (from overlay_cam_on_image()).
      titles: Titles for each subplot.
      figsize: Matplotlib figure size.
      cmap_name: Colormap for the grayscale panel (usually 'gray').
   """
   g = gray_img.numpy() if isinstance(gray_img, tf.Tensor) else gray_img
   g = g.squeeze()

   c = cam.numpy() if isinstance(cam, tf.Tensor) else cam
   c01 = _normalize_array(c)

   plt.figure(figsize=figsize)
   # Panel 1: grayscale
   plt.subplot(1, 3, 1)
   plt.imshow(g, cmap=cmap_name)
   plt.title(titles[0]); plt.axis("off")

   # Panel 2: heatmap (matplotlib cmap)
   plt.subplot(1, 3, 2)
   plt.imshow(c01, cmap="jet")
   plt.title(titles[1]); plt.axis("off")

   # Panel 3: overlay
   plt.subplot(1, 3, 3)
   plt.imshow(overlay_rgb)
   plt.title(titles[2]); plt.axis("off")
   plt.tight_layout()
   plt.show()

def save_triptych_png(
   path: str,
   gray_img: tf.Tensor | np.ndarray,
   cam: tf.Tensor | np.ndarray,
   alpha: float = 0.35,
   gamma: float = 1.0,
   cmap_name: str = "jet",
   figsize: Tuple[int, int] = (12, 4),
) -> None:
   """
   Save a 3-panel PNG (X-ray, heatmap, overlay) to disk.

   Args:
      path (str): Output PNG path.
      gray_img: [H,W] or [H,W,1] image.
      cam: [H,W] Grad-CAM heatmap.
      alpha (float): Overlay opacity.
      gamma (float): Gamma for base.
      cmap_name (str): Colormap for heatmap/overlay.
      figsize (Tuple[int,int]): Figure size.
   """
   overlay_rgb = overlay_cam_on_image(gray_img, cam, alpha, gamma, cmap_name)
   g = gray_img.numpy() if isinstance(gray_img, tf.Tensor) else gray_img
   g = g.squeeze()
   c = cam.numpy() if isinstance(cam, tf.Tensor) else cam
   c01 = _normalize_array(c)

   fig = plt.figure(figsize=figsize)
   ax1 = fig.add_subplot(1, 3, 1)
   ax1.imshow(g, cmap="gray"); ax1.set_title("X-ray"); ax1.axis("off")

   ax2 = fig.add_subplot(1, 3, 2)
   ax2.imshow(c01, cmap=cmap_name); ax2.set_title("Grad-CAM"); ax2.axis("off")

   ax3 = fig.add_subplot(1, 3, 3)
   ax3.imshow(overlay_rgb); ax3.set_title("Overlay"); ax3.axis("off")

   plt.tight_layout()
   fig.savefig(path, dpi=200, bbox_inches="tight")
   plt.close(fig)
   
   
# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def _normalize_array(x: np.ndarray) -> np.ndarray:
   """
   Normalize any array to [0,1]. If constant, returns zeros.

   Args:
      x (np.ndarray): Arbitrary float array.

   Returns:
      np.ndarray: Same shape as input, scaled to [0,1].
   """
   x = x.astype(np.float32)
   mn, mx = np.min(x), np.max(x)
   if mx > mn:
      return (x - mn) / (mx - mn + 1e-8)
   return np.zeros_like(x, dtype=np.float32)
