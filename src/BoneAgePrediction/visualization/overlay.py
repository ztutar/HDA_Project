# src/visualization/overlay.py
from typing import Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def normalize_array(x: np.ndarray) -> np.ndarray:
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
   # Convert to numpy
   g = gray_img.numpy() if isinstance(gray_img, tf.Tensor) else gray_img
   c = cam.numpy() if isinstance(cam, tf.Tensor) else cam
   g = g.squeeze()  # [H,W]
   if g.ndim != 2 or c.ndim != 2:
      raise ValueError("Expected gray_img [H,W] or [H,W,1] and cam [H,W].")

   # Normalize
   g01 = normalize_array(g)
   if gamma != 1.0:
      g01 = np.power(np.clip(g01, 0.0, 1.0), gamma)
   c01 = normalize_array(c)

   # Colorize heatmap 
   cmap = plt.get_cmap(cmap_name)
   cam_rgb01 = cmap(c01)[..., :3].astype(np.float32)  # [H,W,3] in [0,1]

   # Grayscale to 3-ch
   g_rgb01 = np.repeat(g01[..., None], 3, axis=-1)

   # Per-pixel alpha scales with heat intensity
   a = (c01[..., None] * float(alpha)).astype(np.float32)
   out01 = (1.0 - a) * g_rgb01 + a * cam_rgb01
   out01 = np.clip(out01, 0.0, 1.0)
   return (out01 * 255.0).astype(np.uint8)

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
   c01 = normalize_array(c)

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
   c01 = normalize_array(c)

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
