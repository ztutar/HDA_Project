"""Utilities for maintaining the experiment summary CSV.

The functions in this module keep the `results.csv` file up to date by
flattening nested configuration objects, migrating legacy formats, and ensuring
each run appends a consistent row. They are designed to be run repeatedly
without manual cleanup, so they aggressively normalize headers and values.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence
import csv
import json
import os

from BAP.utils.config import (
   DataConfig,
   ModelConfig,
   ProjectConfig,
   ROIConfig,
   TrainingConfig,
)

NA_VALUE = "N/A"
CONFIG_COLUMN_PREFIX = "cfg."

# Order of the summary columns before configuration-specific entries.
SUMMARY_BASE_HEADER: List[str] = [
   "model_name",
   "num_params",
   "total_training_time_s",
   "train_mae",
   "train_rmse",
   "val_mae",
   "val_rmse",
   "test_mae",
   "test_rmse",
   "stopped_epoch",
   "best_epoch",
   "config_file",
   "save_dir",
]

_DEFAULT_CONFIG_KEYS: List[str] | None = None


def _flatten_mapping(
   mapping: Mapping[str, Any],
   parent_key: str = "",
) -> Dict[str, Any]:
   """Recursively flatten nested mappings into dotted-key dictionaries.

   Args:
      mapping:
         Arbitrary mapping that may contain nested dict-like values.
      parent_key:
         Prefix applied to nested keys; used internally during recursion.

   Returns:
      Dict[str, Any]
         A single-level dictionary where `foo.bar` denotes the `bar` key within the
         nested `foo` dictionary.
   """
   items: Dict[str, Any] = {}
   for key, value in mapping.items():
      new_key = f"{parent_key}.{key}" if parent_key else key
      if isinstance(value, Mapping):
         items.update(_flatten_mapping(value, new_key))
      else:
         items[new_key] = value
   return items


def _format_config_value(value: Any) -> str:
   """Normalize configuration values to strings that are safe for CSV storage.

   Lists and dictionaries are JSON-encoded to preserve structure while
   remaining deterministic (sorted keys for dicts). `None` collapses to
   ``NA_VALUE`` so missing information is explicit in the output file.
   """
   if isinstance(value, (dict, list)):
      return json.dumps(value, sort_keys=isinstance(value, dict))
   if value is None:
      return NA_VALUE
   return str(value)


def _get_default_config_keys() -> List[str]:
   """Return the sorted list of configuration keys derived from defaults.

   Instantiates a `ProjectConfig` with default child configs, flattens the
   resulting dataclass tree, and caches the keys so repeated calls avoid
   redundant work.
   """
   global _DEFAULT_CONFIG_KEYS
   if _DEFAULT_CONFIG_KEYS is None:
      defaults = ProjectConfig(
         data=DataConfig(),
         roi=ROIConfig(),
         model=ModelConfig(),
         training=TrainingConfig(),
         raw={},
      )
      defaults_dict = asdict(defaults)
      defaults_dict.pop("raw", None)
      defaults_dict.pop("config_name", None)
      _DEFAULT_CONFIG_KEYS = sorted(_flatten_mapping(defaults_dict).keys())
   return _DEFAULT_CONFIG_KEYS


def _read_existing_header(results_csv: str) -> List[str] | None:
   """Read the header row from an existing summary file, if any.

   Args:
   results_csv:
      Absolute or relative path to the summary CSV.

   Returns:
   list[str] | None
      The header fields if the file exists and contains at least one row,
      otherwise ``None``.
   """
   if not os.path.exists(results_csv):
      return None
   with open(results_csv, newline="") as f:
      reader = csv.reader(f)
      try:
         return next(reader)
      except StopIteration:
         return None


def _migrate_legacy_summary(
   results_csv: str,
   desired_keys: Sequence[str],
) -> List[str]:
   """Expand legacy summary files that store configurations as raw JSON blobs.

   Old summaries kept the entire configuration inside a `config_params` column.
   This function rewrites the file so each flattened configuration key receives
   its own column, merging keys from both legacy rows and the requested
   ``desired_keys``.

   Returns:
   list[str]
      The sorted collection of config keys used in the rewritten file.
   """
   with open(results_csv, newline="") as f:
      reader = csv.DictReader(f)
      rows = list(reader)

   union_keys = set(desired_keys)
   flattened_rows: List[tuple[Dict[str, str], Dict[str, Any]]] = []
   for row in rows:
      raw_json = row.get("config_params", "")
      try:
         raw_config = json.loads(raw_json) if raw_json else {}
      except json.JSONDecodeError:
         raw_config = {}
      flattened = _flatten_mapping(raw_config) if isinstance(raw_config, dict) else {}
      union_keys.update(flattened.keys())
      flattened_rows.append((row, flattened))

   final_keys = sorted(union_keys)
   header = SUMMARY_BASE_HEADER + [CONFIG_COLUMN_PREFIX + key for key in final_keys]

   with open(results_csv, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      for row, flattened in flattened_rows:
         base_row = [row.get(col, NA_VALUE) for col in SUMMARY_BASE_HEADER]
         config_row = [
            _format_config_value(flattened.get(key, NA_VALUE))
            if key in flattened
            else NA_VALUE
            for key in final_keys
         ]
         writer.writerow(base_row + config_row)

   return final_keys


def _rewrite_summary_with_keys(
   results_csv: str,
   config_keys: Sequence[str],
) -> None:
   """Rewrite a summary CSV so it matches the provided configuration columns.

   The base columns remain in the original order, while `config_keys` determine
   the trailing configuration-specific columns. Existing data is re-aligned
   against the new header to keep previous rows intact.
   """
   with open(results_csv, newline="") as f:
      reader = csv.DictReader(f)
      rows = list(reader)

   header = SUMMARY_BASE_HEADER + [CONFIG_COLUMN_PREFIX + key for key in config_keys]
   with open(results_csv, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      for row in rows:
         base_row = [row.get(col, NA_VALUE) for col in SUMMARY_BASE_HEADER]
         config_row = [
            row.get(CONFIG_COLUMN_PREFIX + key, NA_VALUE) or NA_VALUE
            for key in config_keys
         ]
         writer.writerow(base_row + config_row)


def _ensure_summary_structure(
   results_csv: str,
   desired_keys: Sequence[str],
) -> List[str]:
   """Ensure the summary CSV exists and its header covers the desired keys.

   The function creates the file if missing, migrates legacy formats, and
   rewrites inconsistent headers. The returned list reflects the definitive
   order of configuration columns that callers must follow when writing rows.
   """
   dir_name = os.path.dirname(results_csv)
   if dir_name:
      os.makedirs(dir_name, exist_ok=True)

   existing_header = _read_existing_header(results_csv)
   if existing_header is None:
      config_keys = sorted(set(desired_keys))
      header = SUMMARY_BASE_HEADER + [CONFIG_COLUMN_PREFIX + key for key in config_keys]
      with open(results_csv, "w", newline="") as f:
         writer = csv.writer(f)
         writer.writerow(header)
      return config_keys

   if "config_params" in existing_header:
      return _migrate_legacy_summary(results_csv, desired_keys)

   base_len = len(SUMMARY_BASE_HEADER)
   existing_base = existing_header[:base_len]
   existing_config_keys = [
      col[len(CONFIG_COLUMN_PREFIX):]
      for col in existing_header
      if col.startswith(CONFIG_COLUMN_PREFIX)
   ]

   if existing_base != SUMMARY_BASE_HEADER:
      merged_keys = sorted(set(existing_config_keys) | set(desired_keys))
      _rewrite_summary_with_keys(results_csv, merged_keys)
      return merged_keys

   config_keys = list(existing_config_keys)
   for key in desired_keys:
      if key not in config_keys:
         config_keys.append(key)

   if config_keys != existing_config_keys:
      _rewrite_summary_with_keys(results_csv, config_keys)

   return config_keys


def append_summary_row(
   results_csv: str,
   base_data: Mapping[str, Any],
   config_bundle: ProjectConfig,
) -> None:
   """Append a training summary row with normalized configuration values.

   Args:
      results_csv:
         Path to the experiment summary CSV, created if absent.
      base_data:
         Mapping containing the metrics defined by ``SUMMARY_BASE_HEADER``.
      config_bundle:
         `ProjectConfig` instance whose `raw` dict provides the configuration
         values appended after the base metrics.
   """
   flattened_config = _flatten_mapping(config_bundle.raw or {})
   desired_keys = sorted(
      set(_get_default_config_keys()) | set(flattened_config.keys())
   )
   config_keys = _ensure_summary_structure(results_csv, desired_keys)

   base_row = []
   for key in SUMMARY_BASE_HEADER:
      if key == "config_file":
         base_row.append(config_bundle.config_name)
      else:
         value = base_data.get(key, NA_VALUE)
         base_row.append(str(value))

   config_row = [
      _format_config_value(flattened_config.get(key, NA_VALUE))
      if key in flattened_config
      else NA_VALUE
      for key in config_keys
   ]

   with open(results_csv, "a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(base_row + config_row)
