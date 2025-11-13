"""Helper utilities for constructing the standard training callbacks.

This module centralizes the wiring of every callback used when training the
bone age prediction models.
"""

from typing import List
import os
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger


def make_callbacks(save_dir: str, model_name: str = "model", patience: int = 10) -> List[Callback]:
   """Create the default set of Keras callbacks for model training. The function 
   creates the `save_dir` if it does not exist, making it safe to call in fresh 
   experiment folders. The callbacks use validation MAE as the optimization signal 
   because the downstream evaluation focuses on that metric.

   Args:
      save_dir:
         Directory where artifacts such as checkpoints and logs are stored.
      model_name:
         Prefix used to name each saved artifact so multiple experiments can share
         the same parent directory without overwriting one another.
      patience:
         Number of epochs without validation improvement tolerated by the
         `EarlyStopping` callback before training is halted.

   Returns:
      List[Callback]
         A list containing, in order:
            1. `ModelCheckpoint` that stores the best validation-MAE model.
            2. `EarlyStopping` that restores the best weights once patience is
               exhausted.
            3. `TensorBoard` writer for interactive experiment inspection.
            4. `CSVLogger` that appends epoch-level metrics to disk.

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
   
