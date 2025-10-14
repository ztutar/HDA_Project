from typing import Tuple, Optional, Dict
import tensorflow as tf
import numpy as np
from keras import Model, backend, layers

def compute_GradCAM(
         model: Model,
         image: tf.Tensor,
         target_layer_name: Optional[str] = None,
) -> tf.Tensor:
   """
   Compute Grad-CAM heatmap for a single image and a single-output regression model.

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
   x = tf.expand_dims(image, axis=0) # [1,H,W,C]
   
   # Fİnd a conv layer if not given
   if target_layer_name is None:
      for layer in reversed(model.layers):
         if isinstance(layer, layers.Conv2D):
            target_layer_name = layer.name
            break
   
   if target_layer_name is None:
      raise ValueError("No Conv2D layer found for Grad-CAM.")
   
   target_layer = model.get_layer(target_layer_name)
   grad_model = Model(
      inputs=[model.inputs],
      outputs=[target_layer.output, model.output]
   )
   
   with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(x, training=False)
      tape.watch(conv_outputs)
      # For regression, use the scalar output directly
      target = predictions[:, 0]
   
   grads = tape.gradient(target, conv_outputs) # [1,h,w,c]
   weights = tf.reduce_mean(grads, axis=(1,2), keepdims=True) # TODO: add explanation, what does this line do


   
   