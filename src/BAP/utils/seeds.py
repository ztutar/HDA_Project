"""Utility helpers to enforce deterministic behavior throughout the project.

This module centralizes every side effect required to obtain reproducible
experiments with TensorFlow/Keras: seeding Python's hash randomization,
propagating the chosen seed through TensorFlow and Keras helpers, and toggling
TensorFlow's op determinism flag. Importing and calling :func:`set_seeds`
should be the very first step of any training or evaluation script so the
entire execution graph, including dataloaders and augmentation pipelines,
shares the exact same pseudo-random sequence across runs.
"""

import os
import tensorflow as tf
import keras
#from BAP.utils.logger import get_logger  

#logger = get_logger(__name__)

def set_seeds(seed: int = 42):
   """Seed all pseudo-random components used by TensorFlow and Keras.

   The function synchronizes Python's hash randomization with Keras/TensorFlow
   RNGs and enables deterministic kernel execution, which drastically reduces
   run-to-run variance. Call this once per process before constructing models
   or datasets to guarantee exhaustive reproducibility on both CPU and GPU.

   Parameters
   ----------
   seed:
      Integer value applied to every available RNG to guarantee that pseudo-
      random operations (weight initialization, shuffling, augmentation, etc.)
      emit the same results across executions.
   """

   #logger.info("Setting random seeds to %d", seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   keras.utils.set_random_seed(seed)
   tf.keras.utils.set_random_seed(seed)
   tf.config.experimental.enable_op_determinism()
   #logger.debug("Seeds applied to os.environ, random, numpy, tensorflow and keras.")
