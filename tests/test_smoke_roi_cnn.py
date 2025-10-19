"""Smoke test for the ROI CNN training pipeline on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from BoneAgePrediction.training import train_R1 as train_module


def _write_minimal_config(tmp_path: Path) -> Path:
    roi_root = tmp_path / "roi_cache"
    config_dict = {
        "data": {
            "data_path": str(tmp_path / "placeholder-data"),
            "target_h": 64,
            "target_w": 64,
            "keep_aspect_ratio": True,
            "pad_value": 0.0,
            "batch_size": 2,
            "clahe": False,
            "augment": False,
            "cache": False,
        },
        "roi": {
            "locator": {
                "roi_path": str(roi_root),
                "num_blocks": 1,
                "channels": [8],
                "dense_units": 8,
                "epochs": 1,
                "learning_rate": 1e-3,
            },
            "extractor": {
                "roi_size": 32,
                "carpal_margin": 0.1,
                "meta_mask_radius": 0.1,
                "heatmap_threshold": 0.5,
                "save_heatmaps": False,
            },
        },
        "model": {
            "channels": [8],
            "dense_units": 8,
            "use_gender": False,
        },
        "training": {
            "epochs": 1,
            "patience": 1,
            "loss": "mse",
            "huber_delta": 1.0,
            "save_dir": str(tmp_path / "checkpoints"),
            "results_csv": str(tmp_path / "smoke_results.csv"),
            "optimizer": {
                "learning_rate": 1e-3,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-7,
            },
        },
    }
    config_path = tmp_path / "roi_cnn_smoke.json"
    config_path.write_text(json.dumps(config_dict))
    return config_path


def _fake_make_roi_dataset(
    *,
    data_path: str,
    roi_path: str,
    split: str,
    batch_size: int,
    cache: bool,
) -> tf.data.Dataset:
    del data_path, roi_path, cache
    height = 16
    width = 16
    num_samples = max(batch_size * 2, 4)
    carpal = np.random.rand(num_samples, height, width, 1).astype(np.float32)
    metaph = np.random.rand(num_samples, height, width, 1).astype(np.float32)
    carpal_box = np.tile(np.array([1, 2, 3, 4], dtype=np.int32), (num_samples, 1))
    metaph_box = np.tile(np.array([5, 6, 7, 8], dtype=np.int32), (num_samples, 1))
    gender = np.zeros(num_samples, dtype=np.int32)
    image_ids = np.array(
        [f"{split}_sample_{idx}".encode("utf-8") for idx in range(num_samples)],
        dtype="S16",
    )
    ages = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    features = {
        "carpal": carpal,
        "metaph": metaph,
        "carpal_box": carpal_box,
        "metaph_box": metaph_box,
        "gender": gender,
        "image_id": image_ids,
    }
    dataset = tf.data.Dataset.from_tensor_slices((features, ages))
    return dataset.batch(batch_size, drop_remainder=False)


@pytest.mark.smoke
def test_roi_cnn_training_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_minimal_config(tmp_path)
    roi_root = tmp_path / "roi_cache"
    for split in ("train", "validation", "test"):
        (roi_root / split / "carpal").mkdir(parents=True, exist_ok=True)
        (roi_root / split / "metaph").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(train_module, "make_roi_dataset", _fake_make_roi_dataset)
    monkeypatch.setattr(train_module, "train_locator_and_save_rois", lambda *_, **__: None)
    monkeypatch.setattr(train_module, "make_callbacks", lambda *_, **__: [])

    model, history = train_module.train_ROI_CNN(str(config_path))

    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == 1
    assert history.epoch == [0]
    assert "loss" in history.history and len(history.history["loss"]) == 1
    results_path = tmp_path / "smoke_results.csv"
    assert results_path.exists() and results_path.read_text().strip()
