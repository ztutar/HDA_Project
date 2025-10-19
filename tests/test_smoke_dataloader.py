"""Smoke-test coverage for the dataset loader pipeline.

This file checks that every YAML config in `experiments/configs` can build a
TensorFlow dataset without raising errors. It discovers configs, loads each one
into a frozen `ConfigCase`, and parametrizes the single test so every config
is exercised. The helpers resolve project paths, adjust batch settings for the
test run, and bail out early when required resources or optional dependencies
are missing."""


from dataclasses import dataclass
import sys
from pathlib import Path

import pytest
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from BoneAgePrediction.data import dataset_loader
from BoneAgePrediction.data.dataset_loader import make_dataset, make_roi_dataset
from BoneAgePrediction.utils.config import DataConfig, load_config


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


def _write_dummy_png(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    """Create a simple grayscale PNG image for ROI dataset smoke tests."""
    ramp = np.linspace(0, 255, num=size[0] * size[1], dtype=np.uint8).reshape(
        size[0], size[1]
    )
    image = tf.constant(ramp[..., np.newaxis], dtype=tf.uint8)
    encoded = tf.io.encode_png(image)
    path.write_bytes(encoded.numpy())


@dataclass(frozen=True)
class ConfigCase:
    path: Path
    config: DataConfig
    batch_size: int
    target_h: int
    target_w: int
    keep_aspect_ratio: bool
    pad_value: float
    clahe: bool
    augment: bool
    cache: bool
    max_batches: int = 5

    def id_string(self) -> str:
        return (
            f"{self.path.stem}"
            f"[split={self.config.split},"
            f"target={self.target_h}x{self.target_w},"
            f"keep_ar={self.keep_aspect_ratio},"
            f"pad={self.pad_value},"
            f"clahe={self.clahe},"
            f"augment={self.augment},"
            f"batch_size={self.batch_size},"
            f"cache={self.cache},"
            f"num_batches={self.max_batches}]"
        )


def _load_config_cases():
    cases = []
    for cfg_path in _collect_config_paths():
        cfg_bundle = load_config(str(cfg_path))
        cfg = cfg_bundle.data
        batch_size = max(1, min(cfg.batch_size, 8))
        target_h = max(8, min(cfg.target_h, 256))
        target_w = max(8, min(cfg.target_w, 256))
        keep_aspect_ratio = cfg.keep_aspect_ratio
        pad_value = cfg.pad_value
        clahe = cfg.clahe
        augment = cfg.augment if cfg.split.lower() == "train" else False
        cases.append(
            ConfigCase(
                path=cfg_path,
                config=cfg,
                batch_size=batch_size,
                target_h=target_h,
                target_w=target_w,
                keep_aspect_ratio=keep_aspect_ratio,
                pad_value=pad_value,
                clahe=clahe,
                augment=augment,
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

    if case.clahe and not dataset_loader._HAS_CV2:
        pytest.skip("OpenCV not installed; skipping CLAHE config")

    dataset = make_dataset(
        data_path=str(data_root),
        split="train",
        target_h=case.target_h,
        target_w=case.target_w,
        keep_aspect_ratio=case.keep_aspect_ratio,
        pad_value=case.pad_value,
        batch_size=case.batch_size,
        clahe=case.clahe,
        augment=case.augment,
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
        assert images.shape[1] == case.target_h
        assert images.shape[2] == case.target_w
        assert genders.shape[0] == labels.shape[0] > 0
        assert bool(tf.math.reduce_all(tf.math.is_finite(images)).numpy())

    if batch_count < max_batches:
        pytest.skip(
            f"Only {batch_count} batches available for config {config_path.stem}, expected {max_batches}"
        )


def test_make_roi_dataset_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "train").mkdir()
    (data_root / "train.csv").write_text(
        "Image ID,male,Bone Age (months)\n"
        "123,1,150.0\n",
        encoding="utf-8",
    )

    roi_root = tmp_path / "rois"
    carpal_dir = roi_root / "train" / "carpal"
    metaph_dir = roi_root / "train" / "metaph"
    carpal_dir.mkdir(parents=True)
    metaph_dir.mkdir(parents=True)
    _write_dummy_png(carpal_dir / "123.png")
    _write_dummy_png(metaph_dir / "123.png")

    (roi_root / "train" / "roi_coords.csv").write_text(
        "image_id,carpal_y0,carpal_x0,carpal_y1,carpal_x1,metaph_y0,metaph_x0,metaph_y1,metaph_x1\n"
        "123,1,2,5,6,3,4,7,8\n",
        encoding="utf-8",
    )

    dataset = make_roi_dataset(
        data_path=str(data_root),
        roi_path=str(roi_root),
        split="train",
        batch_size=1,
        cache=False,
    )

    features, labels = next(iter(dataset.take(1)))

    carpal = features["carpal"].numpy()
    metaph = features["metaph"].numpy()
    assert carpal.shape == (1, 8, 8, 1)
    assert metaph.shape == (1, 8, 8, 1)
    assert features["gender"].numpy().tolist() == [1]
    assert features["image_id"].numpy().tolist() == [b"123"]
    np.testing.assert_array_equal(
        features["carpal_box"].numpy(), np.array([[1, 2, 5, 6]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        features["metaph_box"].numpy(), np.array([[3, 4, 7, 8]], dtype=np.int32)
    )
    assert labels.numpy().tolist() == [150.0]
