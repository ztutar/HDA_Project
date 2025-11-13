"""Utility helpers for managing experiment output folders and model metadata.

The functions in this module centralize how training runs create disk paths
and persist bookkeeping information:

* ``incremental_path`` guarantees that every run gets its own directory by
   enumerating suffixed folders under a ``model/config`` hierarchy.
* ``load_model_dicts`` restores previously saved metadata (metrics, config
   hashes, checkpoints info, etc.) so downstream scripts can resume or analyze
   experiments without recomputing results.
* ``save_model_dicts`` writes the updated metadata atomically to avoid partial
   files when the process is interrupted, keeping experiment tracking robust.

Using these helpers keeps experiment orchestration deterministic and removes
incidental complexity (manual folder bookkeeping, JSON I/O boilerplate) from
training scripts.
"""

import os
import json
from typing import Any, Dict

def incremental_path(save_dir: str, model_name: str = None, config_name: str = None) -> str:
   """Return a fresh run directory path for the given model/config combination.

   The function builds ``<save_dir>/<model_name>/<config_name>/`` and then
   searches for the first folder named ``{model_name}_{config_name}_{nn}``
   (``nn`` is zero-padded) that does not yet exist. The folder is created on the
   fly and the full path is returned so callers can safely write files without
   clobbering earlier runs.

   Args:
      save_dir: Root directory where all experiment artifacts are stored.
      model_name: High-level identifier for the model architecture or family.
      config_name: Identifier for the configuration or hyper-parameter set.

   Returns:
      The absolute path to a newly created, unique run directory.

   Raises:
      RuntimeError: If 98 folders already exist for the same model/config
      combination, which likely signals a runaway loop creating directories.
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
   """Load serialized model metadata from ``results_path`` if it exists.

   Args:
      results_path: JSON file that stores per-model metadata dictionaries.

   Returns:
      A nested dictionary keyed by model name (outer) and arbitrary metadata
      keys (inner). Returns an empty dict when the file does not exist so
      caller code can treat the absence of prior runs as the default case.
   """
   if not os.path.exists(results_path):
      return {}
   with open(results_path, "r", encoding="utf-8") as fp:
      return json.load(fp)


def save_model_dicts(results: Dict[str, Dict[str, Any]], results_path: str) -> None:
   """Persist the current metadata to disk via an atomic JSON file swap.

   Args:
      results: Nested dictionary produced during training/evaluation.
      results_path: Destination JSON file path.

   The function writes to ``<path>.tmp`` first and then atomically replaces the
   final file. This guards against partial writes (e.g., power loss) that would
   corrupt the metadata store and break subsequent ``load_model_dicts`` calls.
   """
   tmp_path = f"{results_path}.tmp"
   os.makedirs(os.path.dirname(results_path), exist_ok=True)
   with open(tmp_path, "w", encoding="utf-8") as fp:
      json.dump(results, fp, indent=2)
   os.replace(tmp_path, results_path)
