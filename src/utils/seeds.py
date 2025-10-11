"""
This module provides a function to set seeds for various libraries including
Python's built-in random module, NumPy, and TensorFlow.
It also sets the PYTHONHASHSEED environment variable.

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