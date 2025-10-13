
"""Utilities for loading project configuration for data processing workflows.

The project now organises experiment configuration files into top-level
sections such as ``data:``, ``model:``, and ``training:``.  This module defines
lightweight dataclasses for the supported sections and provides helpers to load
them from YAML/JSON files.  The main entry point, :func:`load_config`, returns a
``ProjectConfig`` bundle that exposes the parsed dataclasses while keeping the
raw mapping available for callers that need direct access to other keys.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin
import logging
import yaml
import json
try:
   from BoneAgePrediction.utils.logger import get_logger  # type: ignore
except ImportError:  # fallback to stdlib logger if package not installed
   get_logger = logging.getLogger  # type: ignore

CONFIG_BASE_DIR = Path("experiments/configs")

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

@dataclass
class ModelConfig:
   num_blocks: int = 3
   channels: list[int] = field(default_factory=lambda: [32, 64, 128])
   dense_units: int = 64

@dataclass
class OptimizerConfig:
   learning_rate: float = 0.0003
   beta_1: float = 0.9
   beta_2: float = 0.999
   epsilon: float = 1e-7

@dataclass
class TrainingConfig:
   epochs: int = 30
   patience: int = 10  # Early stopping patience
   loss: str = "huber"
   huber_delta: float = 10.0
   save_dir: str = "experiments/checkpoints"
   results_csv: str = "experiments/train_results_summary.csv"
   optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

@dataclass
class ProjectConfig:
   data: DataConfig
   model: ModelConfig
   training: TrainingConfig
   raw: Dict[str, Any]


def _ensure_mapping(section: Any, section_name: str, path_obj: Path) -> Dict[str, Any]:
   if section is None:
      return {}
   if not isinstance(section, dict):
      raise ValueError(
         f"Section '{section_name}' in {path_obj} must be a mapping, got {type(section).__name__}."
      )
   return section


def _coerce_to_field_type(value: Any, field_type: Any) -> Any:
   origin = get_origin(field_type)
   if origin in (list, tuple) and isinstance(value, (list, tuple)):
      element_types = get_args(field_type) or (Any,)
      element_type = element_types[0] if element_types else Any
      return [
         _coerce_to_field_type(item, element_type) for item in value
      ]
   if isinstance(value, str):
      lowered = value.strip().lower()
      if field_type is bool:
         if lowered in {"true", "1", "yes", "y"}:
            return True
         if lowered in {"false", "0", "no", "n"}:
            return False
      if field_type is float:
         try:
            return float(value)
         except ValueError:
            return value
      if field_type is int:
         try:
            return int(value)
         except ValueError:
            return value
   return value


def _filter_known_fields(
   section: Dict[str, Any],
   dataclass_type: type,
   section_name: str,
   logger: logging.Logger,
   path_obj: Path,
) -> Dict[str, Any]:
   field_map = {dataclass_field.name: dataclass_field for dataclass_field in fields(dataclass_type)}
   filtered_section = {}
   for key, value in section.items():
      if key in field_map:
         filtered_section[key] = _coerce_to_field_type(value, field_map[key].type)
   ignored_keys = set(section) - set(field_map)
   if ignored_keys:
      logger.warning(
         "Ignoring unsupported %s config keys in %s: %s",
         section_name,
         path_obj,
         ", ".join(sorted(ignored_keys)),
      )
   return filtered_section


def load_config(path: Optional[str] = None) -> ProjectConfig:
   """
   Load configuration from a YAML or JSON file.
   Args:
      path (Optional[str]): Path to the configuration file. If None, returns defaults.
   Returns:
      ProjectConfig: Bundle containing parsed data/model/training sections and the raw mapping.
   """
   logger = get_logger(__name__)
   if path is None:
      logger.info("No config path provided; using defaults.")
      return ProjectConfig(
         data=DataConfig(),
         model=ModelConfig(),
         training=TrainingConfig(),
         raw={},
      )
   path_obj = Path(path)
   if not path_obj.is_absolute():
      if path_obj.parent != CONFIG_BASE_DIR and CONFIG_BASE_DIR not in path_obj.parents:
         path_obj = CONFIG_BASE_DIR / path_obj
   if path_obj.suffix in {".yaml", ".yml"}:
      logger.info("Loading YAML config: %s", path_obj)
      with path_obj.open("r") as f:
         config_dict = yaml.safe_load(f)
   else:
      logger.info("Loading JSON config: %s", path_obj)
      with path_obj.open("r") as f:
         config_dict = json.load(f)
   logger.debug("Config loaded: %s", config_dict)
   if config_dict is None:
      config_dict = {}
   if not isinstance(config_dict, dict):
      raise ValueError(f"Configuration at {path_obj} must be a mapping.")

   data_section = _ensure_mapping(config_dict.get("data", {}), "data", path_obj)
   model_section = _ensure_mapping(config_dict.get("model", {}), "model", path_obj)
   training_section = _ensure_mapping(config_dict.get("training", {}), "training", path_obj)

   data_config = DataConfig(
      **_filter_known_fields(data_section, DataConfig, "data", logger, path_obj)
   )
   model_config = ModelConfig(
      **_filter_known_fields(model_section, ModelConfig, "model", logger, path_obj)
   )

   optimizer_section = training_section.get("optimizer", {})
   optimizer_mapping = _ensure_mapping(optimizer_section, "training.optimizer", path_obj)
   optimizer_config = OptimizerConfig(
      **_filter_known_fields(
         optimizer_mapping,
         OptimizerConfig,
         "training.optimizer",
         logger,
         path_obj,
      )
   )

   # Remove the optimizer section before filtering training keys to avoid redundant warnings.
   training_section_without_optimizer = {
      key: value for key, value in training_section.items() if key != "optimizer"
   }
   training_filtered = _filter_known_fields(
      training_section_without_optimizer, TrainingConfig, "training", logger, path_obj
   )
   training_config = TrainingConfig(**training_filtered, optimizer=optimizer_config)

   return ProjectConfig(
      data=data_config,
      model=model_config,
      training=training_config,
      raw=config_dict,
   )
