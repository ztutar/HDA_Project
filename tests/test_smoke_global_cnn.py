"""Smoke test for the GlobalCNN training pipeline on synthetic data."""

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

from BoneAgePrediction.training import train_B0 as train_module


def _write_minimal_config(tmp_path: Path) -> Path:
    config_dict = {
        "data": {
            "data_path": str(tmp_path / "placeholder-data"),
            "target_h": 32,
            "target_w": 32,
            "keep_aspect_ratio": True,
            "pad_value": 0.0,
            "batch_size": 2,
            "clahe": False,
            "augment": False,
            "cache": False,
        },
        "model": {
            "num_blocks": 1,
            "channels": [8],
            "dense_units": 8,
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
    config_path = tmp_path / "global_cnn_smoke.json"
    config_path.write_text(json.dumps(config_dict))
    return config_path


def _synthetic_dataset(target_h: int, target_w: int, batch_size: int) -> tf.data.Dataset:
    num_samples = max(batch_size * 2, 4)
    images = np.random.rand(num_samples, target_h, target_w, 1).astype(np.float32)
    genders = np.zeros(num_samples, dtype=np.int32)
    ages = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"image": images, "gender": genders}, ages)
    )
    return dataset.batch(batch_size, drop_remainder=False)


def _fake_make_dataset(
    *,
    data_path: str,
    split: str,
    target_h: int,
    target_w: int,
    keep_aspect_ratio: bool,
    pad_value: float,
    batch_size: int,
    clahe: bool,
    augment: bool,
    cache: bool,
) -> tf.data.Dataset:
    del (
        data_path,
        split,
        keep_aspect_ratio,
        pad_value,
        clahe,
        augment,
        cache,
    )
    return _synthetic_dataset(target_h, target_w, batch_size)


@pytest.mark.smoke
def test_global_cnn_training_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_minimal_config(tmp_path)
    monkeypatch.setattr(train_module, "make_dataset", _fake_make_dataset)

    model, history = train_module.train_GlobalCNN(str(config_path))

    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == 1
    assert history.epoch == [0]
    assert "loss" in history.history and len(history.history["loss"]) == 1
    results_path = tmp_path / "smoke_results.csv"
    assert results_path.exists() and results_path.read_text().strip()
