#TODO: add docstring explanation for this module. write in details and explain each function


import tensorflow as tf
from keras import Model, layers
import numpy as np


def compute_GradCAM(
         model: Model,
         image: tf.Tensor,
) -> tf.Tensor:
   """
   Compute a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap for
   a single input according to the canonical Keras implementation.

   Args:
      model (keras.Model): Trained Keras model mapping [B,H,W,C] -> [B,K].
      image (tf.Tensor): Image tensor of shape [H,W,C], dtype float32.
      target_layer_name (Optional[str]): Name of the last conv layer. If None,
         automatically pick the last Conv2D layer.
      target_index (Optional[int]): Index of the output neuron to explain. If None,
         uses the argmax for multi-output models or 0 for single-output models.

   Returns:
      tf.Tensor: Heatmap tensor of shape [H,W] with values scaled to [0,1].
   """

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

   
def compute_GradCAM_plus(
         model: Model,
         image: tf.Tensor,
) -> tf.Tensor:
   """
   Compute a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap for
   a single input according to the canonical Keras implementation.

   Args:
      model (keras.Model): Trained Keras model mapping [B,H,W,C] -> [B,K].
      image (tf.Tensor): Image tensor of shape [H,W,C], dtype float32.
      target_layer_name (Optional[str]): Name of the last conv layer. If None,
         automatically pick the last Conv2D layer.
      target_index (Optional[int]): Index of the output neuron to explain. If None,
         uses the argmax for multi-output models or 0 for single-output models.

   Returns:
      tf.Tensor: Heatmap tensor of shape [H,W] with values scaled to [0,1].
   """

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

   heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
   heatmap = np.maximum(heatmap, 0)
   max_heat = np.max(heatmap)
   if max_heat == 0:
      max_heat = 1e-10
   heatmap /= max_heat
   
   model_hw = tf.shape(image_for_model)[0:2]
   heatmap = tf.image.resize(heatmap[..., tf.newaxis], size=model_hw, method="bilinear")
   heatmap = tf.squeeze(heatmap, axis=-1)

   heatmap = tf.image.resize(heatmap[..., tf.newaxis], size=orig_hw, method="bilinear")
   heatmap = tf.squeeze(heatmap, axis=-1)

   return heatmap