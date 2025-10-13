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
   random.seed(seed)
   np.random.seed(seed)
   tf.random.set_seed(seed)
   logger.debug("Seeds applied to os.environ, random, numpy, and tensorflow.")
