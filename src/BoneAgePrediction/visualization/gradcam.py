from typing import Tuple, Optional, Dict
import tensorflow as tf
import numpy as np
from keras import Model, backend, layers
from BoneAgePrediction.visualization.overlay import overlay_cam_on_image


def compute_GradCAM(
         model: Model,
         image: tf.Tensor,
         target_layer_name: Optional[str] = None,
) -> tf.Tensor:
   """
   Compute Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap 
   for a single image and a single-output regression model. 
   Grad-CAM, helps to know which parts of the feature map contributed most 
   to the model’s final prediction.

   Args:
      model (keras.Model): Trained Keras model mapping [B,H,W,C] -> [B,1].
      image (tf.Tensor): Image tensor shape [H,W,C], float32.
      target_layer_name (Optional[str]): Name of the last conv layer. If None,
         use the last conv-like layer automatically.

   Returns:
      tf.Tensor: Heatmap in [H,W] range [0,1], float32.
   """
   
   if image.ndim != 3:
      raise ValueError("image must be [H,W,C]")
   x = tf.expand_dims(image, axis=0) # -> [1, H, W, C] add batch dim for Keras
   
   # If a conv layer not given, walk layers from the end and
   #  pick the last Conv2D (best spatial-semantic tradeoff).
   if target_layer_name is None:
      for layer in reversed(model.layers):
         if isinstance(layer, layers.Conv2D):
            target_layer_name = layer.name
            break
         
   if target_layer_name is None:
      raise ValueError("No Conv2D layer found for Grad-CAM.")
   
   # Build a sub-model that outputs (feature_maps_of_target_layer, model_output)
   target_layer = model.get_layer(target_layer_name)
   grad_model = Model(
      inputs=[model.inputs],
      outputs=[target_layer.output, model.output]
   )

   # Record operations in order to take gradients of the scalar output wrt conv features
   with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(x, training=False)   # conv_out: [1,h,w,c], preds: [1,1]
      tape.watch(conv_outputs)                                    # tell tape to track conv_out
      # For regression, use the scalar output directly
      target = predictions[:, 0]                                  # regression scalar y (age months)
   
   # Compute the gradient of the model's output wrt the feature maps of some conv layer.
   # d(target)/d(conv_out): gradient tensor with same spatial & channel dims as conv_out
   grads = tape.gradient(target, conv_outputs) # [1,H,W,C] - one gradient map per channel
   
   # For each channel in the conv layer, compute a single importance weight
   # This is a global avg pooling (GAP) of the gradient over the spatial dimensions (H, W).
   # These weights tell us how strongly each feature map channel influenced the output.
   weights = tf.reduce_mean(grads, axis=(1,2), keepdims=True) # [1,1,1,C]
   
   # Weighted combination of feature maps (per-channel) and keep only positive evidence
   heatmap = tf.nn.relu(tf.reduce_sum(weights * conv_outputs, axis=-1)) # [1,H,W] after sum over C
   heatmap = tf.squeeze(heatmap, axis=0) # [H,W]
   
   # Normalize the heatmap to [0,1] for visualization stability
   heatmap = heatmap - tf.reduce_min(heatmap)
   denom = tf.reduce_max(heatmap)
   heatmap = heatmap / (denom + 1e-8)
   
   # Resize heatmap back to input spatial size using bilinear interpolation
   heatmap = tf.image.resize(heatmap[..., None], size=tf.shape(x)[1:3], method = "bilinear")
   heatmap = tf.squeeze(heatmap, axis=-1) # [H,W]
   return heatmap

def compute_GradCAM_overlay(
   model: tf.keras.Model,
   img: tf.Tensor,
   target_layer_name: Optional[str] = None,
   alpha: float = 0.35,
   gamma: float = 1.0,
   cmap_name: str = "jet") -> np.ndarray:
   """
   Compute Grad-CAM and return an RGB overlay using a colormap.

   Args:
      model: Trained regression model [B,H,W,C] -> [B,1].
      img: Single image [H,W,C] float32.
      target_layer_name: Conv layer to probe (auto-select if None).
      alpha: Overlay opacity.
      gamma: Gamma for base grayscale.
      cmap_name: Matplotlib colormap, e.g., 'jet', 'magma', 'turbo'.

   Returns:
      np.ndarray: [H,W,3] uint8 overlay ready to save or display.
   """
   cam = compute_GradCAM(model, img, target_layer_name=target_layer_name)   # [H,W]
   base = img[..., 0] if img.shape[-1] == 1 else tf.image.rgb_to_grayscale(img)[..., 0]
   return overlay_cam_on_image(base, cam, alpha=alpha, gamma=gamma, cmap_name=cmap_name)
   