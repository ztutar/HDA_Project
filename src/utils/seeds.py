"""Utility helpers for keeping experiments repeatable.

This module contains a single helper function, `set_seeds`, that applies the
same seed value across the Python standard library, NumPy, and TensorFlow. By
calling it before running experiments or training models, we ensure that pseudo
random number generators behave consistently across runs, which makes debugging
and comparing results easier.
"""


import os
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed: int = 42):
    """
    Set seeds for reproducibility across various libraries.
    Args:
        seed (int): The seed value to set. 
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
