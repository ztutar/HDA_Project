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
    batch_size: int
    shuffle_buffer: int
    num_workers: int
    cache: bool
    max_batches: int = 5

    def id_string(self) -> str:
        return (
            f"{self.path.stem}"
            f"[split={self.config.split},"
            f"target={self.config.target_h}x{self.config.target_w},"
            f"clahe={self.config.clahe},"
            f"augment={self.config.augment},"
            f"batch_size={self.batch_size},"
            f"shuffle_buffer={self.shuffle_buffer},"
            f"num_workers={self.num_workers},"
            f"cache={self.cache},"
            f"num_batches={self.max_batches}]"
        )


def _load_config_cases():
    cases = []
    for cfg_path in _collect_config_paths():
        cfg = load_config(str(cfg_path))
        batch_size = min(8, cfg.batch_size)
        shuffle_buffer = max(1, min(cfg.shuffle_buffer, 64))
        cases.append(
            ConfigCase(
                path=cfg_path,
                config=cfg,
                batch_size=batch_size,
                shuffle_buffer=shuffle_buffer,
                num_workers=1,
                cache=False,
            )
        )
    return cases


CONFIG_CASES = _load_config_cases()
if not CONFIG_CASES:
    pytest.skip("No non-empty configs under experiments/configs", allow_module_level=True)


@pytest.mark.parametrize(
    "case",
    CONFIG_CASES,
    ids=lambda case: case.id_string(),
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
        batch_size=case.batch_size,
        shuffle_buffer=case.shuffle_buffer,
        num_workers=case.num_workers,
        clahe=cfg.clahe,
        augment=cfg.augment,
        cache=case.cache,
    )

    max_batches = case.max_batches
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
