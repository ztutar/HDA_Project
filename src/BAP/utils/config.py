
"""Utilities for loading project configuration for data processing workflows.

The project now organises experiment configuration files into top-level
sections such as ``data:``, ``roi:``, ``model:``, and ``training:``.  This
module defines lightweight dataclasses for the supported sections and provides
helpers to load them from YAML/JSON files.  The main entry point,
:func:`load_config`, returns a ``ProjectConfig`` bundle that exposes the parsed
dataclasses while keeping the raw mapping available for callers that need
direct access to other keys.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, get_args, get_origin
import logging
import yaml
from BAP.utils.logger import get_logger

logger = get_logger(__name__)


CONFIG_BASE_DIR = Path("experiments/configs")

@dataclass
class DataConfig:
   data_path: str = "data/raw"            # Default path to raw data
   image_size: int = 512                  # Target image size (square)
   keep_aspect_ratio: bool = True         # Whether to keep aspect ratio when resizing
   batch_size: int = 16                   # Default batch size
   clahe: bool = False                    # Whether to apply CLAHE
   augment: bool = False                  # Whether to apply data augmentation

@dataclass
class ModelConfig:
   global_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
   global_dense_units: int = 128
   roi_channels: list[int] = field(default_factory=lambda: [32, 64])
   roi_dense_units: int = 32
   fusion_dense_units: list[int] = field(default_factory=lambda: [256, 128])
   use_gender: bool = False
   dropout_rate: float = 0.2

@dataclass
class TrainingConfig:
   epochs: int = 30
   patience: int = 10  # Early stopping patience
   learning_rate: float = 0.0003
   loss: str = "huber"
   results_csv: str = "experiments/train_results_summary.csv"
   perform_test: bool = False

@dataclass
class ROILocatorConfig:
   roi_path: str = "data/processed/cropped_rois"
   pretrained_model_path: str = ""

@dataclass
class ROIExtractorConfig:
   roi_size: int = 256
   heatmap_threshold: float = 0.4
   save_heatmaps: bool = False

@dataclass
class ROIConfig:
   locator: ROILocatorConfig = field(default_factory=ROILocatorConfig)
   extractor: ROIExtractorConfig = field(default_factory=ROIExtractorConfig)

@dataclass
class ProjectConfig:
   data: DataConfig = field(default_factory=DataConfig)
   roi: ROIConfig = field(default_factory=ROIConfig)
   model: ModelConfig = field(default_factory=ModelConfig)
   training: TrainingConfig = field(default_factory=TrainingConfig)
   config_name: str = "default"
   raw: Dict[str, Any] = field(default_factory=dict)
   
#TODO: Add EvaluationConfig dataclass and include it in ProjectConfig


def load_config(path: Optional[str] = None) -> ProjectConfig:
   """
   Load configuration from a YAML or use defaults.
   Args:
      path (Optional[str]): Path to the configuration file. If None, returns defaults.
   Returns:
      ProjectConfig: Bundle containing parsed data/model/training sections and the raw mapping.
   """
   if path is None:
      logger.info("No config path provided; using defaults.")
      return ProjectConfig(
         data=DataConfig(),
         roi=ROIConfig(),
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
         logger.info("Config loaded: %s", config_dict)
   except Exception as e:
      logger.info("Failed to load config, using defaults.")
      raise ValueError(f"Failed to load config file {path_obj}: {e}") from e
   
   if config_dict is None:
      config_dict = {}
      logger.info("Config file %s is empty; using defaults.", path_obj)
      
   if not isinstance(config_dict, dict):
      raise ValueError(f"Configuration at {path_obj} must be a mapping.")

   # Data Section
   data_section = _ensure_mapping(config_dict.get("data", {}), "data", path_obj)
   data_section = _filter_known_fields(data_section, DataConfig, "data", logger, path_obj)
   data_config = DataConfig(**data_section)
   
   # ROI Section
   roi_section = _ensure_mapping(config_dict.get("roi", {}), "roi", path_obj)
   roi_locator_section = _ensure_mapping(roi_section.get("locator", {}), "roi.locator", path_obj)
   roi_locator_section = _filter_known_fields(
      roi_locator_section, ROILocatorConfig, "roi.locator", logger, path_obj)
   roi_locator_config = ROILocatorConfig(**roi_locator_section)
   
   roi_extractor_section = _ensure_mapping(roi_section.get("extractor", {}), "roi.extractor", path_obj)
   roi_extractor_section = _filter_known_fields(
      roi_extractor_section, ROIExtractorConfig, "roi.extractor", logger, path_obj)
   roi_extractor_config = ROIExtractorConfig(**roi_extractor_section)
   
   roi_config = ROIConfig(locator=roi_locator_config, extractor=roi_extractor_config)
   
   # Model Section
   model_section = _ensure_mapping(config_dict.get("model", {}), "model", path_obj)
   model_section = _filter_known_fields(model_section, ModelConfig, "model", logger, path_obj)
   model_config = ModelConfig(**model_section)
   
   # Training Section
   training_section = _ensure_mapping(config_dict.get("training", {}), "training", path_obj)
   training_section = _filter_known_fields(training_section, TrainingConfig, "training", logger, path_obj)
   training_config = TrainingConfig(**training_section)

   # Test/Evaluation Section
   #TODO: add evaluation section 
   
   logger.info("Configuration loaded successfully from %s", path_obj)

   return ProjectConfig(
      data=data_config,
      roi=roi_config,
      model=model_config,
      training=training_config,
      config_name=str(path_obj),
      raw=config_dict,
   )
   
# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

# Ensure that a section is a mapping; raise ValueError if not.
def _ensure_mapping(section: Any, section_name: str, path_obj: Path) -> Dict[str, Any]:
   if section is None:
      return {}
   if not isinstance(section, dict):
      raise ValueError(
         f"Section '{section_name}' in {path_obj} must be a mapping, got {type(section).__name__}."
      )
   return section

# Filter a section to only include known fields of the dataclass type.
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

# Coerce a value to the expected field type, handling basic conversions.
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
