"""
This module provides utilities for managing file paths and persisting model metadata.

It includes functions to create incremental paths and save/load model dictionaries.
"""

import os
import json
from typing import Any, Dict


def incremental_path(save_dir: str, model_name: str = None, config_name: str = None) -> str:
   """
   Create an incremental path for saving model results.

   Parameters
   ----------
   save_dir : str
      Base save directory.
   model_name : str
      Name of the model.
   config_name : str
      Name of the configuration.

   Returns
   -------
   str
      Unique incremental path.
   """
   # Define the top-level folder based on the save_dir and configuration name.
   head_folder = os.path.join(save_dir, model_name, config_name)
   os.makedirs(head_folder, exist_ok=True)  # Ensure the top-level folder exists.

   # Loop to find a unique folder name by appending an incremental number.
   for n in range(1, 99):
      save_folder = os.path.join(head_folder, f"{model_name}_{config_name}_{n:02d}")  # Construct folder name with zero padding.
      if not os.path.exists(save_folder):  # Check if the folder already exists.
         os.makedirs(save_folder)  # Create the folder if it doesn't exist.
         return save_folder  # Return the unique folder path.

   # If the loop exceeds the limit, raise an error (unlikely in practice).
   raise RuntimeError(f"Too many folders created for {config_name}")


# Utility helpers for persisting model metadata between sessions
def load_model_dicts(results_path: str) -> Dict[str, Dict[str, Any]]:
   """
   Load model dictionaries from a JSON file.

   Parameters
   ----------
   results_path : str
      Path to the JSON file.

   Returns
   -------
   Dict[str, Dict[str, Any]]
      Loaded model dictionaries.
   """
   if not os.path.exists(results_path):
      return {}
   with open(results_path, "r", encoding="utf-8") as fp:
      return json.load(fp)


def save_model_dicts(results: Dict[str, Dict[str, Any]], results_path: str) -> None:
   """
   Save model dictionaries to a JSON file.

   Parameters
   ----------
   results : Dict[str, Dict[str, Any]]
      Model dictionaries to save.
   results_path : str
      Path to the JSON file.
   """
   tmp_path = f"{results_path}.tmp"
   os.makedirs(os.path.dirname(results_path), exist_ok=True)
   with open(tmp_path, "w", encoding="utf-8") as fp:
      json.dump(results, fp, indent=2)
   os.replace(tmp_path, results_path)
