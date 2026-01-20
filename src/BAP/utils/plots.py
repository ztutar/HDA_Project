"""
This module provides plotting and visualization utilities for the bone age prediction project.

It includes functions to display images, plot distributions, and compare model metrics.
"""

import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from typing import Any, Dict, Sequence, Union
from IPython.display import display
from BAP.utils.dataset_loader import load_image_original, load_image_grayscale, apply_clahe

def display_sample_images(metadata: pd.DataFrame, image_dir: Path, n_samples: int=4) -> None:
   """
   Display sample images from the dataset with metadata.

   Parameters
   ----------
   metadata : pd.DataFrame
      DataFrame with image metadata.
   image_dir : Path
      Directory containing images.
   n_samples : int
      Number of samples to display.
   """
   sample_metadata = metadata.sample(n_samples)
   plt.figure(figsize=(15, 5))
   for i, (idx, row) in enumerate(sample_metadata.iterrows()):
      image_id = row['Image ID']
      image_path = os.path.join(image_dir, f'{image_id}.png')
      image = load_image_original(image_path)
      plt.subplot(1, n_samples, i + 1)
      plt.imshow(image)
      boneage = row['Bone Age (months)'] 
      plt.title(f"Image ID: {image_id}\nBone Age: {boneage} months\nGender: {'Male' if row['male'] else 'Female'}")
      plt.axis('off')
   plt.show()
   
   
def plot_distributions(metadata: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
   """
   Plot distributions of bone age and gender for dataset splits.

   Parameters
   ----------
   metadata : Union[pd.DataFrame, Dict[str, pd.DataFrame]]
      Metadata DataFrame or dictionary of splits.
   """
   # Accept legacy single-DataFrame input by wrapping it in a dict
   if isinstance(metadata, dict):
      splits: Dict[str, pd.DataFrame] = metadata
   else:
      splits = {"All": metadata}

   if not splits:
      raise ValueError("metadata must contain at least one split/DataFrame.")

   split_names = list(splits.keys())
   num_splits = len(split_names)

   fig, axes = plt.subplots(2, num_splits, figsize=(5 * num_splits + 2, 8))
   axes = np.atleast_2d(axes)

   for col_idx, split_name in enumerate(split_names):
      df = splits[split_name]
      bone_ax = axes[0, col_idx]
      gender_ax = axes[1, col_idx]

      sns.histplot(df["Bone Age (months)"], kde=True, bins=30, ax=bone_ax)
      bone_ax.set_title(f"{split_name} Bone Age")
      bone_ax.set_xlabel("Bone Age (months)")
      bone_ax.set_ylabel("Frequency")

      sns.countplot(x="male", data=df, ax=gender_ax)
      gender_ax.set_title(f"{split_name} Gender Distribution")
      gender_ax.set_xlabel("Gender")
      gender_ax.set_ylabel("Count")
      gender_ax.set_xticks(ticks=[0, 1])
      gender_ax.set_xticklabels(["Female", "Male"])

   fig.suptitle("Dataset Distributions by Split", fontsize=14)
   plt.tight_layout(rect=[0, 0, 1, 0.96])
   plt.show()


def display_test_predictions(
   metadata: pd.DataFrame,
   predictions: Union[Sequence[float], np.ndarray],
   image_dir: Path,
   n_samples: int = 8,
   seed: int | None = None,
) -> None:
   """
   Display test predictions with true vs predicted ages.

   Parameters
   ----------
   metadata : pd.DataFrame
      Test metadata.
   predictions : Union[Sequence[float], np.ndarray]
      Predicted ages.
   image_dir : Path
      Directory containing images.
   n_samples : int
      Number of samples to display.
   seed : int | None
      Random seed.
   """
   predictions_arr = np.asarray(predictions).reshape(-1)
   if len(predictions_arr) != len(metadata):
      raise ValueError(
         f"Predictions length ({len(predictions_arr)}) does not match metadata length ({len(metadata)})."
      )

   df = metadata.reset_index(drop=True).copy()
   df["Predicted Bone Age (months)"] = predictions_arr

   sample_count = min(n_samples, len(df))
   sample_df = df.sample(sample_count, random_state=seed)

   num_cols = min(4, sample_count)
   num_rows = int(np.ceil(sample_count / num_cols))
   fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 4.25 * num_rows))
   axes = np.atleast_1d(axes).ravel()

   for ax, (_, row) in zip(axes, sample_df.iterrows()):
      image_id = row["Image ID"]
      image_path = Path(image_dir) / f"{image_id}.png"
      if not image_path.exists():
         ax.set_title(f"Missing image ID {image_id}")
         ax.axis("off")
         continue

      image = load_image_original(str(image_path)).numpy()
      ax.imshow(image)
      true_age = row["Bone Age (months)"]
      pred_age = row["Predicted Bone Age (months)"]
      ax.set_title(f"ID {image_id}\nTrue: {true_age:.1f}m | Pred: {pred_age:.1f}m", fontsize=12)
      ax.axis("off")

   for ax in axes[sample_count:]:
      ax.axis("off")

   fig.suptitle("Test Samples — True vs Predicted Bone Age", fontsize=14)
   plt.tight_layout(rect=[0, 0, 1, 0.95])
   plt.show()

# Raw vs. CLAHE enhanced images
def display_raw_vs_clahe_images(metadata: pd.DataFrame, image_dir: Path) -> None:
   """
   Display raw vs CLAHE enhanced images.

   Parameters
   ----------
   metadata : pd.DataFrame
      Metadata DataFrame.
   image_dir : Path
      Directory containing images.
   """
   sample_row = metadata.sample(1).iloc[0]
   image_id = str(sample_row["Image ID"])
   image_path = Path(image_dir) / f"{image_id}.png"
   if not image_path.exists():
      raise FileNotFoundError(f"Image not found at {image_path}")

   raw_img = load_image_grayscale(str(image_path))
   clahe_img = apply_clahe(raw_img)

   raw_img_np = tf.squeeze(raw_img, axis=-1).numpy()
   clahe_img_np = tf.squeeze(clahe_img, axis=-1).numpy()
   difference = np.clip(clahe_img_np - raw_img_np, -1.0, 1.0)

   fig, axes = plt.subplots(1, 3, figsize=(10, 4))
   axes[0].imshow(raw_img_np, cmap="gray")
   axes[0].set_title(f"Raw | ID {image_id}")
   axes[0].axis("off")

   axes[1].imshow(clahe_img_np, cmap="gray")
   axes[1].set_title("CLAHE Applied")
   axes[1].axis("off")

   axes[2].imshow(difference, cmap="bwr", vmin=-0.2, vmax=0.2)
   axes[2].set_title("CLAHE - Raw")
   axes[2].axis("off")

   fig.suptitle("Raw vs CLAHE contrast enhancement", y=1.02)
   plt.tight_layout()
   plt.show()


def plot_training_metrics(metrics_dict: Dict, model_name="Model"):
   """
   Plot training metrics for a model.

   Parameters
   ----------
   metrics_dict : Dict
      Metrics dictionary.
   model_name : str
      Name of the model.
   """
   history = metrics_dict['history']
   times_per_epoch = metrics_dict.get('times_per_epoch')

   epochs = range(1, len(history['mae']) + 1)

   fig, axs = plt.subplots(1, 3 if times_per_epoch is not None else 2, figsize=(14, 5))

   # MAE Plot
   axs[0].plot(epochs, history['mae'], label='Train MAE')
   axs[0].plot(epochs, history['val_mae'], label='Val MAE')
   axs[0].set_title(f'{model_name} MAE per Epoch')
   axs[0].set_xlabel('Epoch')
   axs[0].set_ylabel('MAE')
   axs[0].legend()
   axs[0].grid(True)

   # Loss Plot
   axs[1].plot(epochs, history['loss'], label='Train Loss')
   axs[1].plot(epochs, history['val_loss'], label='Val Loss')
   axs[1].set_title(f'{model_name} Loss per Epoch')
   axs[1].set_xlabel('Epoch')
   axs[1].set_ylabel('Loss')
   axs[1].legend()
   axs[1].grid(True)

   plt.tight_layout()
   plt.show()


def training_metrics_table(metrics: Dict[str, Any], model_name: str) -> None:
   """
   Display training metrics in a table.

   Parameters
   ----------
   metrics : Dict[str, Any]
      Metrics dictionary.
   model_name : str
      Name of the model.
   """
   if not isinstance(metrics, dict) or not metrics:
      raise ValueError("metrics must be provided as a non-empty dictionary.")

   splits = ("train", "val", "test")
   metric_names = ("loss", "mae", "rmse")
   has_required = any(f"{split}_{metric}" in metrics for split in splits for metric in metric_names)
   if not has_required:
      raise ValueError("metrics dictionary does not include the required split keys.")

   rows = []
   for split in splits:
      row = {"Split": split.title()}
      for metric in metric_names:
         key = f"{split}_{metric}"
         value = metrics.get(key)
         if value is None:
            row[metric.upper()] = None
         elif isinstance(value, (int, float, np.floating)):
            row[metric.upper()] = float(value)
         else:
            row[metric.upper()] = value
      rows.append(row)

   df = pd.DataFrame(rows).set_index("Split")
   styled = df.style.set_caption(f"{model_name} Training Metrics").format(precision=4, na_rep="—")
   display(styled)


def compare_models_table(
   model_results_dict: Dict[str, Dict[str, Any]],
   model_metrics_dict: Dict[str, Dict[str, Any]],
) -> None:
   """
   Compare models in a table.

   Parameters
   ----------
   model_results_dict : Dict[str, Dict[str, Any]]
      Model results dictionary.
   model_metrics_dict : Dict[str, Dict[str, Any]]
      Model metrics dictionary.
   """
   model_names = sorted(set(model_results_dict.keys()) | set(model_metrics_dict.keys()))
   if not model_names:
      raise ValueError("No models found in the supplied dictionaries.")

   metrics_fields = [
      ("train_loss", "Train Loss"),
      ("train_mae", "Train MAE"),
      ("train_rmse", "Train RMSE"),
      ("val_loss", "Val Loss"),
      ("val_mae", "Val MAE"),
      ("val_rmse", "Val RMSE"),
      ("test_loss", "Test Loss"),
      ("test_mae", "Test MAE"),
      ("test_rmse", "Test RMSE"),
   ]
   result_fields = [
      ("num_params", "Params"),
      ("training_time", "Training Time (s)"),
      ("num_epochs_ran", "Epochs Ran"),
      ("best_epoch_idx", "Best Epoch Idx"),
   ]

   rows = []
   for name in model_names:
      metrics = model_metrics_dict.get(name, {})
      results = model_results_dict.get(name, {})
      row = {"Model": name}

      for key, label in metrics_fields:
         value = metrics.get(key)
         if isinstance(value, (int, float, np.floating)):
            row[label] = float(value)
         else:
            row[label] = value

      for key, label in result_fields:
         value = results.get(key)
         if isinstance(value, (np.integer,)) or isinstance(value, int):
            row[label] = int(value)
         elif isinstance(value, (float, np.floating)):
            row[label] = float(value)
         else:
            row[label] = value

      rows.append(row)

   df = pd.DataFrame(rows).set_index("Model")
   float_cols = [label for _, label in metrics_fields] + ["Training Time (s)"]
   format_dict = {col: "{:.4f}" for col in float_cols if col in df.columns}
   styled = df.style.set_caption("Model Comparison Metrics & Training Stats").format(format_dict, na_rep="—")
   display(styled)


def compare_training_metrics(model_metrics_dict: Dict[str, Dict]) -> None:
   """
   Compare training metrics across models.

   Parameters
   ----------
   model_metrics_dict : Dict[str, Dict]
      Model metrics dictionary.
   """
   if not model_metrics_dict:
      raise ValueError("model_metrics_dict must contain at least one model entry.")

   fig, ax = plt.subplots(figsize=(8, 5))

   for model_name, metrics in model_metrics_dict.items():
      history = metrics.get("history")
      if history is None:
         raise ValueError(f"Missing 'history' for model '{model_name}'.")

      val_mae = history.get("val_mae")

      if val_mae:
         linestyle = "-" if "_Gender" in model_name else "--"
         ax.plot(
            range(1, len(val_mae) + 1),
            val_mae,
            linestyle=linestyle,
            label=f"{model_name} Val",
         )

   ax.set_title("MAE per Epoch (All Models)")
   ax.set_xlabel("Epoch")
   ax.set_ylabel("MAE")
   ax.set_ylim(top=60)
   ax.grid(True, linestyle=":")
   ax.legend()

   plt.tight_layout()
   plt.show()


def plot_best_epoch_metrics(
   model_metrics_dict: Dict[str, Dict],
   model_results_dict: Dict[str, Dict],
   splits: Sequence[str] = ("train", "val", "test"),
) -> None:
   """
   Plot best epoch metrics for models.

   Parameters
   ----------
   model_metrics_dict : Dict[str, Dict]
       Model metrics dictionary.
   model_results_dict : Dict[str, Dict]
       Model results dictionary.
   splits : Sequence[str]
       Splits to plot.
   """
   if not model_metrics_dict:
      raise ValueError("model_metrics_dict must contain at least one model entry.")

   model_names = [name for name in model_metrics_dict.keys() if name in model_results_dict]
   if not model_names:
      raise ValueError("No overlapping models found between metrics and results dictionaries.")

   num_models = len(model_names)
   num_splits = len(splits)
   x = np.arange(num_models)
   width = 0.8 / max(num_splits, 1)
   palette = sns.color_palette("tab10", n_colors=num_splits)

   fig, ax = plt.subplots(figsize=(8, 5))

   split_labels = {s: s.title() for s in splits}

   for idx, split in enumerate(splits):
      key = f"{split}_mae"
      offsets = x + (idx - (num_splits - 1) / 2) * width
      values = []
      for name in model_names:
         value = model_metrics_dict[name].get(key)
         values.append(value if value is not None else np.nan)
      bars = ax.bar(offsets, values, width=width, color=palette[idx], label=f"{split_labels.get(split, split)} MAE")

      for bar_idx, bar in enumerate(bars):
         value = values[bar_idx]
         if value is None or not np.isfinite(value):
            continue
         model_name = model_names[bar_idx]
         ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
         )

   ax.set_title("MAE at Best Epoch")
   ax.set_ylabel("MAE")
   ax.set_xticks(x)
   ax.set_xticklabels(model_names, rotation=20, ha="right")
   ax.grid(axis="y", linestyle=":", alpha=0.4)
   ax.legend()

   fig.suptitle("Model Comparisons - MAE per Split", fontsize=14)
   plt.tight_layout(rect=[0, 0, 1, 0.94])
   plt.show()


