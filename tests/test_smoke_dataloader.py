from dataclasses import dataclass
import sys
from pathlib import Path

import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import dataset_loader
from src.data.dataset_loader import make_dataset
from src.utils.config import DataConfig, load_config


CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs"


def _resolve_data_root(data_path: str) -> Path:
    root = Path(data_path).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def _collect_config_paths():
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        cfg for cfg in CONFIG_DIR.glob("*.yaml") if cfg.stat().st_size > 0
    )


@dataclass(frozen=True)
class ConfigCase:
    path: Path
    config: DataConfig


def _load_config_cases():
    cases = []
    for cfg_path in _collect_config_paths():
        cfg = load_config(str(cfg_path))
        cases.append(ConfigCase(path=cfg_path, config=cfg))
    return cases


CONFIG_CASES = _load_config_cases()
if not CONFIG_CASES:
    pytest.skip("No non-empty configs under experiments/configs", allow_module_level=True)


@pytest.mark.parametrize(
    "case",
    CONFIG_CASES,
    ids=lambda case: (
        f"{case.path.stem}"
        f"[split={case.config.split},"
        f"target={case.config.target_h}x{case.config.target_w},"
        f"clahe={case.config.clahe},"
        f"augment={case.config.augment},"
        f"batch={case.config.batch_size}]"
    ),
)
def test_dataloader_smoke(case: ConfigCase):
    cfg = case.config
    config_path = case.path
    data_root = _resolve_data_root(cfg.data_path)
    split_name = {"val": "validation", "validation": "validation"}.get(
        cfg.split.lower(), cfg.split.lower()
    )
    image_dir = data_root / split_name
    csv_file = data_root / f"{split_name}.csv"

    if not image_dir.exists() or not csv_file.exists():
        pytest.skip(f"Split '{split_name}' not available under {data_root}")

    if cfg.clahe and not dataset_loader._HAS_CV2:
        pytest.skip("OpenCV not installed; skipping CLAHE config")

    dataset = make_dataset(
        data_path=str(data_root),
        split=cfg.split,
        target_h=cfg.target_h,
        target_w=cfg.target_w,
        keep_aspect_ratio=cfg.keep_aspect_ratio,
        pad_value=cfg.pad_value,
        batch_size=cfg.batch_size,
        shuffle_buffer=max(1, min(cfg.shuffle_buffer, 64)),
        num_workers=1,
        clahe=cfg.clahe,
        augment=cfg.augment,
        cache=False,
    )

    max_batches = 5
    batch_count = 0
    for batch_count, (features, labels) in enumerate(dataset.take(max_batches), start=1):
        images = features["image"]
        genders = features["gender"]

        assert images.dtype == tf.float32
        assert genders.dtype == tf.int32
        assert labels.dtype == tf.float32
        assert images.shape[-1] == cfg.channels
        assert images.shape[1] == cfg.target_h
        assert images.shape[2] == cfg.target_w
        assert genders.shape[0] == labels.shape[0] > 0
        assert bool(tf.math.reduce_all(tf.math.is_finite(images)).numpy())

    if batch_count < max_batches:
        pytest.skip(
            f"Only {batch_count} batches available for config {config_path.stem}, expected {max_batches}"
        )
