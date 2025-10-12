"""
Smoke test for the data loader pipeline.

This test loads a tiny batch from each split (train/validation/test)
and verifies basic invariants:
- Dataset can be constructed without errors
- First batch can be iterated
- Feature/label keys exist with expected dtypes and shapes
- No NaNs present in tensors

Run with pytest:
  pytest -q tests/test_smoke_dataloader.py

Or directly as a script:
  python -m tests.test_smoke_dataloader
"""

from typing import Tuple
import os
import sys
import numpy as np
import tensorflow as tf


# Support running both as a module and directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset_loader import make_dataset  # type: ignore
from src.utils.config import DataConfig  # type: ignore


def _take_one(ds: tf.data.Dataset) -> Tuple[dict, tf.Tensor]:
    for features, labels in ds.take(1):
        return features, labels
    raise AssertionError("Dataset is empty; no batches yielded.")


def _assert_basic_batch(features: dict, labels: tf.Tensor, cfg: DataConfig) -> None:
    assert isinstance(features, dict), "Features must be a dict with keys 'image' and 'gender'."
    assert "image" in features and "gender" in features, "Missing 'image' or 'gender' in features."

    image = features["image"]
    gender = features["gender"]

    # Dtypes
    assert image.dtype == tf.float32, f"image dtype should be float32, got {image.dtype}"
    assert gender.dtype.is_integer, f"gender dtype should be integer, got {gender.dtype}"
    assert labels.dtype == tf.float32, f"labels dtype should be float32, got {labels.dtype}"

    # Shapes: batch x H x W x 1
    img_shape = image.shape
    assert img_shape.rank == 4, f"image rank should be 4, got {img_shape}"
    assert img_shape[-1] == 1, f"image channels should be 1, got {img_shape[-1]}"
    if cfg.keep_aspect_ratio:
        # We letterbox to exact target size when keep_aspect_ratio=True
        assert img_shape[-3] == cfg.target_h, f"image height should be {cfg.target_h}, got {img_shape[-3]}"
        assert img_shape[-2] == cfg.target_w, f"image width should be {cfg.target_w}, got {img_shape[-2]}"

    # Values: check no NaNs
    for t, name in [(image, "image"), (tf.cast(gender, tf.float32), "gender"), (labels, "labels")]:
        n_nans = int(tf.math.count_nonzero(tf.math.is_nan(tf.cast(t, tf.float32))))
        assert n_nans == 0, f"Found NaNs in {name}: {n_nans}"

    # Gender values should be 0/1
    g_unique = np.unique(gender.numpy())
    assert set(g_unique.tolist()).issubset({0, 1}), f"gender contains values outside {{0,1}}: {g_unique}"


def _build_dataset(cfg: DataConfig, split: str) -> tf.data.Dataset:
    return make_dataset(
        data_path=cfg.data_path,
        split=split,
        target_h=cfg.target_h,
        target_w=cfg.target_w,
        keep_aspect_ratio=cfg.keep_aspect_ratio,
        pad_value=cfg.pad_value,
        batch_size=min(4, cfg.batch_size),  # keep small for speed
        shuffle_buffer=min(256, cfg.shuffle_buffer),
        num_workers=max(1, cfg.num_workers),
        clahe=False,     # keep off for speed/portability
        augment=False,   # deterministic and faster for smoke
        cache=False,     # avoid memory cache for smoke
    )


def test_smoke_loader_train_val_test():
    cfg = DataConfig(
        data_path="data/raw",
        split="train",
        target_h=512,
        target_w=512,
        keep_aspect_ratio=True,
        pad_value=0.0,
        batch_size=4,
        shuffle_buffer=256,
        num_workers=2,
        clahe=False,
        augment=False,
        cache=False,
    )

    for split in ("train", "validation", "test"):
        ds = _build_dataset(cfg, split)
        features, labels = _take_one(ds)
        _assert_basic_batch(features, labels, cfg)


if __name__ == "__main__":
    # Allow quick manual run without pytest
    print("Running data loader smoke test...", flush=True)
    try:
        test_smoke_loader_train_val_test()
        print("OK: smoke test passed.")
    except AssertionError as e:
        print(f"FAILED: {e}")
        raise
