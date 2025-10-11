"""
This module provides a configuration class for data handling and a function to load configurations from a file.
It supports both YAML and JSON formats for configuration files.
The DataConfig class includes default values for various parameters related to data processing.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
import json

@dataclass
class DataConfig:
   """
   Configuration class for data handling parameters. 
   Attributes:
       data_path (str): Path to the raw data.
       split (str): Data split type ('train', 'val', 'test').
       image_size (int): Size of the images.
       batch_size (int): Batch size for data loading.
       shuffle_buffer (int): Buffer size for shuffling data.
       num_workers (int): Number of workers for data loading.
       clahe (bool): Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
       augment (bool): Whether to apply data augmentation.
       cache (bool): Whether to cache the dataset.
       channels (int): Number of image channels.
   """
   data_path: str = "data/raw"            # Default path to raw data
   split: str = "train"                   # Options: 'train', 'val', 'test'
   target_h: int = 512                    # Target image height
   target_w: int = 512                    # Target image width
   keep_aspect_ratio: bool = True         # Whether to keep aspect ratio when resizing
   pad_value: float = 0.0                 # Padding value when resizing with aspect ratio
   batch_size: int = 16                   # Default batch size
   shuffle_buffer: int = 1024             # Default shuffle buffer size
   num_workers: int = 4                   # Default number of workers for data loading
   clahe: bool = False                    # Whether to apply CLAHE
   augment: bool = False                  # Whether to apply data augmentation
   cache: bool = True                     # Whether to cache the dataset
   channels: int = 1                      # Number of image channels
   
def load_config(path: Optional[str] = None) -> DataConfig:
   """
   Load configuration from a YAML or JSON file and return a DataConfig instance.
   Args:
       path (Optional[str]): Path to the configuration file. If None, returns default DataConfig.
   Returns:
       DataConfig: An instance of DataConfig populated with values from the file or defaults.
   """
   if path is None:
      return DataConfig()
   if path.endswith('.yaml') or path.endswith('.yml'):
      with open(path, "r") as f:
         config_dict = yaml.safe_load(f)
   else:
      with open(path, "r") as f:
         config_dict = json.load(f)
   return DataConfig(**config_dict)