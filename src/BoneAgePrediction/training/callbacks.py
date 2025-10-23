
"""
Provide a helper for constructing the standard callback suite used during model
training. The module focuses on assembling TensorFlow Keras callbacks that
handle saving the best checkpoint, stopping training early when validation
metrics stall, and recording logs for TensorBoard visualization. The primary
entry point is `make_callbacks`, which prepares the filesystem paths, creates
each callback object with the project’s preferred settings, and returns them as
an ordered list ready to pass into `model.fit`.
"""

from typing import List
import os
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger


def make_callbacks(save_dir: str, model_name: str = "model", patience: int = 10) -> List[Callback]:
   """
   Creates a list of Keras callbacks for training: Checkpoint, EarlyStopping, TensorBoard 

   Args:
      save_dir (str): Directory to save model checkpoints and logs.
      model_name (str, optional): Used to name the checkpoint file (e.g., "b0_global", "r1_roi").
      patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.

   Returns:
      List[tf.keras.callbacks.Callback]: List of callbacks to pass into model.fit().
   """
   os.makedirs(save_dir, exist_ok=True)
   checkpoint_path = os.path.join(save_dir, f"{model_name}_best.keras")
   tb_logs_path = os.path.join(save_dir, f"{model_name}_tensorboard_logs")
   csv_logs_path = os.path.join(save_dir, f"{model_name}_training_log.csv")
   
   checkpoint_cb = ModelCheckpoint(
      filepath=checkpoint_path,
      monitor='val_mae',
      mode='min',
      save_best_only=True,
      save_weights_only=False,
      verbose=1
   )
   
   earlystop_cb = EarlyStopping(
      monitor='val_mae',
      mode='min',
      patience=patience,
      restore_best_weights=True,
      verbose=1
   )
   
   tensorboard_cb = TensorBoard(
      log_dir=tb_logs_path,
      histogram_freq=1,
      write_graph=False
   )
   
   csvlogger_cb = CSVLogger(
      filename=csv_logs_path,
      separator=",",
      append=True
      )
   
   return [checkpoint_cb, earlystop_cb, tensorboard_cb, csvlogger_cb]
   
