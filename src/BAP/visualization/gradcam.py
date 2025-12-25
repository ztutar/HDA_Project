#TODO: add docstring explanation for this module and it's functions. Write in details.

import tensorflow as tf
from keras import Model, layers
import numpy as np
import matplotlib.cm as cm


def compute_GradCAM(
         model: Model,
         image: tf.Tensor,
) -> tf.Tensor:

   image = tf.convert_to_tensor(image, dtype=tf.float32)
   if image.shape.rank == 2:
      image = image[..., tf.newaxis]
   if image.shape.rank != 3:
      raise ValueError("image must be [H,W,C] or [H,W].")

   orig_hw = tf.shape(image)[0:2]

   input_shape = model.input_shape
   if isinstance(input_shape, list):
      image_input_shape = next((shape for shape in input_shape if len(shape) == 4), None)
   else:
      image_input_shape = input_shape

   if not image_input_shape or len(image_input_shape) != 4:
      raise ValueError("Grad-CAM supports models with a single 4D image input.")

   target_h, target_w, target_c = image_input_shape[1:]

   if target_c and target_c != image.shape[-1]:
      if target_c == 1 and image.shape[-1] == 3:
         image = tf.image.rgb_to_grayscale(image)
      elif target_c == 3 and image.shape[-1] == 1:
         image = tf.image.grayscale_to_rgb(image)
      else:
         raise ValueError(f"Cannot adapt image channels from {image.shape[-1]} to {target_c}.")

   if target_h and target_w:
      image_for_model = tf.image.resize(image, size=(target_h, target_w), method="bilinear")
   else:
      image_for_model = image

   img_batch = tf.expand_dims(image_for_model, axis=0)


   for layer in reversed(model.layers):
      if isinstance(layer, layers.Conv2D):
         target_layer_name = layer.name
         break

   target_layer = model.get_layer(target_layer_name)
   grad_model = Model(
      inputs=model.inputs,
      outputs=[target_layer.output, model.output],
   )

   with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(img_batch, training=False)
      if predictions.shape[-1] == 1:
         target = predictions[:, 0]
      else:
         dynamic_index = tf.argmax(predictions[0])
         target = tf.gather(predictions, dynamic_index, axis=1)

   grads = tape.gradient(target, conv_outputs)
   if grads is None:
      raise ValueError("Could not compute gradients for Grad-CAM.")

   pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
   conv_outputs = conv_outputs[0]

   heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
   heatmap = tf.nn.relu(heatmap)

   max_val = tf.reduce_max(heatmap)
   if tf.equal(max_val, 0):
      heatmap = tf.zeros_like(heatmap)
   else:
      heatmap /= max_val

   model_hw = tf.shape(image_for_model)[0:2]
   heatmap = tf.image.resize(heatmap[..., tf.newaxis], size=model_hw, method="bilinear")
   heatmap = tf.squeeze(heatmap, axis=-1)

   heatmap = tf.image.resize(heatmap[..., tf.newaxis], size=orig_hw, method="bilinear")
   heatmap = tf.squeeze(heatmap, axis=-1)

   return heatmap

   
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
