"""
This module provides utilities for setting random seeds to ensure reproducibility.

It includes functions to set seeds for Python, TensorFlow, and Keras.
"""

import os
import tensorflow as tf
import keras
#from BAP.utils.logger import get_logger  

#logger = get_logger(__name__)


def set_seeds(seed: int = 42):
   """
   Set random seeds for reproducibility.

   Parameters
   ----------
   seed : int
      The seed value to use.
   """
   #logger.info("Setting random seeds to %d", seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   keras.utils.set_random_seed(seed)
   tf.keras.utils.set_random_seed(seed)
   tf.config.experimental.enable_op_determinism()
   #logger.debug("Seeds applied to os.environ, random, numpy, tensorflow and keras.")
