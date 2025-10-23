"""Utility helpers for keeping experiments repeatable.

This module contains a single helper function, `set_seeds`, that applies the
same seed value across the Python standard library, NumPy, and TensorFlow. By
calling it before running experiments or training models, we ensure that pseudo
random number generators behave consistently across runs, which makes debugging
and comparing results easier.
"""


import logging
import os
import random
import numpy as np
import tensorflow as tf
import keras

try:
   from BoneAgePrediction.utils.logger import get_logger  # type: ignore
except ImportError:
   get_logger = logging.getLogger  # type: ignore

logger = get_logger(__name__)

def set_seeds(seed: int = 42):
   """
   Set seeds for reproducibility across various libraries.
   Args:
      seed (int): The seed value to set. 
   """
   logger.info("Setting random seeds to %d", seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   keras.utils.set_random_seed(seed)
   tf.keras.utils.set_random_seed(seed)
   tf.config.experimental.enable_op_determinism()
   logger.debug("Seeds applied to os.environ, random, numpy, tensorflow and keras.")
