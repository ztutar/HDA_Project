"""Utilities for loading and validating experiment configuration files.

The module defines dataclasses that hold defaults for data, ROI processing,
model, and training parameters. The `load_config` entrypoint reads a YAML file,
merges it with the defaults, coerces values to the expected types, and returns a
`ProjectConfig` object that keeps both the structured view and the original raw
mapping so callers can inspect untouched keys if needed. Helper utilities
sanity-check the incoming dictionary by discarding unknown fields and converting
string inputs to numbers, booleans, or sequences whenever possible to keep the
runtime configuration consistent across different sources (CLI, YAML, API).
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
   results_csv: str = "experiments/train_results_summary.csv"
   perform_test: bool = False

@dataclass
class ROILocatorConfig:
   roi_path: str = "data/cropped_rois"
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
   

def load_config(path: Optional[str] = None) -> ProjectConfig:
   """Load a configuration file and return a fully populated `ProjectConfig`.

   Parameters
   ----------
   path:
      Optional path to a YAML configuration file. When omitted, defaults defined
      in the dataclasses are used. Relative paths are resolved under
      `experiments/configs`.

   Returns
   -------
   ProjectConfig
      Structured configuration ready for consumption by training or inference
      pipelines. Contains both typed sections and the original raw dictionary.

   Raises
   ------
   ValueError
      If the file cannot be read, the YAML cannot be parsed, or the top-level
      structure is not a mapping.
   """
   if path is None:
      #logger.info("No config path provided; using defaults.")
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
   
   # ROI Section
   roi_section = config_dict.get("roi", {})
   roi_locator_section = roi_section.get("locator", {})
   roi_locator_section = _filter_known_fields(roi_locator_section, ROILocatorConfig)
   roi_locator_config = ROILocatorConfig(**roi_locator_section)
   
   roi_extractor_section = roi_section.get("extractor", {})
   roi_extractor_section = _filter_known_fields(roi_extractor_section, ROIExtractorConfig)
   roi_extractor_config = ROIExtractorConfig(**roi_extractor_section)
   
   roi_config = ROIConfig(locator=roi_locator_config, extractor=roi_extractor_config)
   
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
      roi=roi_config,
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
   """Return a copy of `section` containing only keys defined on `dataclass_type`.

   The function also coerces values to the field's annotated type so downstream
   dataclass construction receives clean inputs. Unknown keys are ignored
   silently, which shields the loader from typos or future fields that older
   code does not yet understand.
   """
   field_map = {dataclass_field.name: dataclass_field for dataclass_field in fields(dataclass_type)}
   filtered_section = {}
   for key, value in section.items():
      if key in field_map:
         filtered_section[key] = _coerce_to_field_type(value, field_map[key].type)
   return filtered_section

# Coerce a value to the expected field type, handling basic conversions.
def _coerce_to_field_type(value: Any, field_type: Any) -> Any:
   """Coerce `value` to `field_type` when obvious conversions are available.

   Handles common string-to-bool/float/int conversions, as well as recursive
   coercion of homogeneous lists/tuples. Values that cannot be converted are
   returned unchanged so the caller can decide how to handle the mismatch.
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
