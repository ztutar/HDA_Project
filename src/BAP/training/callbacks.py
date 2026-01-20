"""
This module provides utilities for creating Keras callbacks used during model training.
It includes functions to set up common callbacks like model checkpointing, early stopping,
TensorBoard logging, and CSV logging for bone age prediction models.
"""

from typing import List
import os
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger


def make_callbacks(save_dir: str, model_name: str = "model", patience: int = 10) -> List[Callback]:
   """
   Creates a list of Keras callbacks for model training.
   
   This function sets up common callbacks including model checkpointing, early stopping,
   TensorBoard logging, and CSV logging.
   
   Args:
      save_dir (str): Directory path where logs and checkpoints will be saved.
      model_name (str): Name prefix for saved files. Default is "model".
      patience (int): Number of epochs with no improvement after which training will be stopped. Default is 10.
   
   Returns:
      List[Callback]: A list of configured Keras callbacks.
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
      min_delta=0,
      mode='min',
      patience=patience,
      restore_best_weights=True,
      verbose=1
   )
   
   tensorboard_cb = TensorBoard(
      log_dir=tb_logs_path,
      histogram_freq=1,
      write_graph=True
   )
   
   csvlogger_cb = CSVLogger(
      filename=csv_logs_path,
      separator=",",
      append=True
      )
   
   return [checkpoint_cb, earlystop_cb, tensorboard_cb, csvlogger_cb]
   
