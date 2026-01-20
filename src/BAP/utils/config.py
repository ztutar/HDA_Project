"""
This module provides configuration management for the bone age prediction project.

It includes dataclasses for different configuration sections and functions to load and process configuration files.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin
import yaml
#from BAP.utils.logger import get_logger

#logger = get_logger(__name__)

CONFIG_BASE_DIR = Path("experiments/configs")

@dataclass
class DataConfig:
   image_size: int = 512                  # Target image size (square)
   clahe: bool = False                    # Whether to apply CLAHE
   augment: bool = False                  # Whether to apply data augmentation
   batch_size: int = 16                   # Default batch size

@dataclass
class ModelConfig:
   channels: list[int] = field(default_factory=lambda: [32, 64, 128])
   dense_units: int = 128
   base_filters: int = 32                    # Stem filters
   block_filters: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
   blocks_per_stage: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
   num_a_blocks: int = 2              # Number of Inception and InSkipCon A blocks
   num_b_blocks: int = 3              # Number of Inception and InSkipCon B blocks
   num_c_blocks: int = 1              # Number of Inception and InSkipCon C blocks
   scale_a: float = 0.17
   scale_b: float = 0.1
   scale_c: float = 0.2
   use_gender: bool = False
   dropout_rate: float = 0.2

@dataclass
class TrainingConfig:
   epochs: int = 30
   patience: int = 10  # Early stopping patience
   learning_rate: float = 0.0003
   results_csv: str = "experiments/train_results_summary.csv"
   perform_test: bool = False

@dataclass
class ProjectConfig:
   data: DataConfig = field(default_factory=DataConfig)
   model: ModelConfig = field(default_factory=ModelConfig)
   training: TrainingConfig = field(default_factory=TrainingConfig)
   config_name: str = "default"
   raw: Dict[str, Any] = field(default_factory=dict)
   

def load_config(path: Optional[str] = None) -> ProjectConfig:
   """
   Load a project configuration from a YAML file or use defaults.

   Parameters
   ----------
   path : Optional[str]
      Path to the YAML configuration file. If None, uses default configurations.

   Returns
   -------
   ProjectConfig
      The loaded project configuration object.
   """
   if path is None:
      #logger.info("No config path provided; using defaults.")
      return ProjectConfig(
         data=DataConfig(),
         model=ModelConfig(),
         training=TrainingConfig(),
         config_name="default",
         raw={},
      )
   path_obj = Path(path)
   if not path_obj.is_absolute():
      if path_obj.parent != CONFIG_BASE_DIR and CONFIG_BASE_DIR not in path_obj.parents:
         path_obj = CONFIG_BASE_DIR / path_obj   
   try:
      with path_obj.open("r") as f:
         config_dict = yaml.safe_load(f)
         #logger.info("Config loaded: %s", config_dict)
   except Exception as e:
      #logger.info("Failed to load config, using defaults.")
      raise ValueError(f"Failed to load config file {path_obj}: {e}") from e
   
   if config_dict is None:
      config_dict = {}
      #logger.info("Config file %s is empty; using defaults.", path_obj)
      
   if not isinstance(config_dict, dict):
      raise ValueError(f"Configuration at {path_obj} must be a mapping.")

   # Data Section
   data_section = config_dict.get("data", {})
   data_section = _filter_known_fields(data_section, DataConfig)
   data_config = DataConfig(**data_section)
   
   # Model Section
   model_section = config_dict.get("model", {})
   model_section = _filter_known_fields(model_section, ModelConfig)
   model_config = ModelConfig(**model_section)
   
   # Training Section
   training_section = config_dict.get("training", {})
   training_section = _filter_known_fields(training_section, TrainingConfig)
   training_config = TrainingConfig(**training_section)

   
   #logger.info("Configuration loaded successfully from %s", path_obj)

   return ProjectConfig(
      data=data_config,
      model=model_config,
      training=training_config,
      config_name=str(path_obj),
      raw=config_dict,
   )
   
# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

# Filter a section to only include known fields of the dataclass type.
def _filter_known_fields(section: Dict[str, Any], dataclass_type: type) -> Dict[str, Any]:
   """
   Filter a configuration section to only include known fields of the dataclass type.

   Parameters
   ----------
   section : Dict[str, Any]
      The configuration section dictionary.
   dataclass_type : type
      The dataclass type to filter against.

   Returns
   -------
   Dict[str, Any]
      The filtered dictionary with only known fields.
   """
   field_map = {dataclass_field.name: dataclass_field for dataclass_field in fields(dataclass_type)}
   filtered_section = {}
   for key, value in section.items():
      if key in field_map:
         filtered_section[key] = _coerce_to_field_type(value, field_map[key].type)
   return filtered_section

# Coerce a value to the expected field type, handling basic conversions.
def _coerce_to_field_type(value: Any, field_type: Any) -> Any:
   """
   Coerce a value to the expected field type, handling basic conversions.

   Parameters
   ----------
   value : Any
      The value to coerce.
   field_type : Any
      The expected type.

   Returns
   -------
   Any
      The coerced value.
   """
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
