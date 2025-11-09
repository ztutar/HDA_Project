import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from typing import Dict, Sequence, Union
from BAP.utils.dataset_loader import load_image_original, load_image_grayscale, apply_clahe


# Function to visualize sample images from the dataset
def display_sample_images(metadata: pd.DataFrame, image_dir: Path, n_samples: int=4) -> None:
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
   
   
# Function to plot distribution of bone age and gender
def plot_distributions(metadata: pd.DataFrame) -> None:
   plt.figure(figsize=(10, 4))

   # Distribution of Bone Age
   plt.subplot(1, 2, 1)
   sns.histplot(metadata["Bone Age (months)"], kde=True, bins=30)
   plt.title("Distribution of Bone Age")
   plt.xlabel("Bone Age (months)")
   plt.ylabel("Frequency")

   # Distribution of Gender
   plt.subplot(1, 2, 2)
   sns.countplot(x="male", data=metadata)
   plt.title("Distribution of Gender")
   plt.xlabel("Gender")
   plt.ylabel("Count")
   plt.xticks(ticks=[0, 1], labels=["Female", "Male"])

   plt.tight_layout()
   plt.show()


def display_test_predictions(
   metadata: pd.DataFrame,
   predictions: Union[Sequence[float], np.ndarray],
   image_dir: Path,
   n_samples: int = 8,
   seed: int | None = None,
) -> None:
   """
   Display random examples from the test set alongside their predicted ages.

   Args:
      metadata (pd.DataFrame): Test split metadata with `Image ID` and `Bone Age (months)`.
      predictions (Sequence[float] | np.ndarray): Model predictions aligned with `metadata`.
      image_dir (Path): Directory that contains the corresponding `.png` files.
      n_samples (int): Number of random samples to display.
      seed (int | None): Optional random seed for reproducibility.
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
   fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
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
   
   
def display_image_and_rois(
   metadata: pd.DataFrame,
   image_dir: Path,
   roi_dirs: Dict[str, Path],
) -> None:
   """
   Display a random sample with its Grad-CAM heatmap and cropped ROIs.

   Args:
      metadata (pd.DataFrame): Split metadata containing `Image ID`.
      image_dir (Path): Directory containing the original `.png` images.
      roi_dirs (Dict[str, Path]): Mapping with `carpal`, `metaph`, and `heatmaps` directories.
      seed (int | None): Optional seed for reproducible sampling.
   """

   required_columns = {"Image ID"}
   missing = required_columns - set(metadata.columns)
   if missing:
      raise ValueError(f"metadata missing required columns: {', '.join(sorted(missing))}")
   for key in ("carpal", "metaph", "heatmaps"):
      if key not in roi_dirs:
         raise ValueError(f"roi_dirs must include '{key}' path.")

   sample_row = metadata.sample(1).iloc[0]
   image_id = str(sample_row["Image ID"])

   full_image_path = Path(image_dir) / f"{image_id}.png"
   heatmap_path = Path(roi_dirs["heatmaps"]) / f"{image_id}.png"
   carpal_path = Path(roi_dirs["carpal"]) / f"{image_id}.png"
   metaph_path = Path(roi_dirs["metaph"]) / f"{image_id}.png"

   missing_paths = [
      p for p in (full_image_path, heatmap_path, carpal_path, metaph_path) if not p.exists()
   ]
   if missing_paths:
      missing_str = ", ".join(str(p) for p in missing_paths)
      raise FileNotFoundError(f"Missing files for image ID {image_id}: {missing_str}")

   full_image = load_image_original(str(full_image_path)).numpy()
   heatmap_img = load_image_original(str(heatmap_path)).numpy()
   carpal_roi = tf.squeeze(load_image_grayscale(str(carpal_path)), axis=-1).numpy()
   metaph_roi = tf.squeeze(load_image_grayscale(str(metaph_path)), axis=-1).numpy()

   fig, axes = plt.subplots(1, 4, figsize=(16, 4))
   axes[0].imshow(full_image)
   axes[0].set_title(f"Raw | ID {image_id}")
   axes[0].axis("off")

   axes[1].imshow(heatmap_img)
   axes[1].set_title("Grad-CAM Heatmap")
   axes[1].axis("off")

   axes[2].imshow(carpal_roi, cmap="gray")
   axes[2].set_title("Carpal ROI")
   axes[2].axis("off")

   axes[3].imshow(metaph_roi, cmap="gray")
   axes[3].set_title("Metacarpal ROI")
   axes[3].axis("off")

   fig.suptitle("Random Image with Heatmap and Cropped ROIs", fontsize=14)
   plt.tight_layout()
   plt.show()


# Raw vs. CLAHE enhanced images
def display_raw_vs_clahe_images(metadata: pd.DataFrame, image_dir: Path) -> None:
   """
   Visualize a single random image and the effect of CLAHE preprocessing.

   Args:
      metadata (pd.DataFrame): Split metadata including `Image ID`.
      image_dir (Path): Directory containing the original `.png` files.
      seed (int | None): Optional seed to make the sampled image reproducible.
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


def compare_training_metrics(model_metrics_dict: Dict[str, Dict]) -> None:
   """
   Plot MAE and loss curves for all models in `model_metrics_dict` for side-by-side comparison.

   Args:
      model_metrics_dict (Dict[str, Dict]): Mapping of model names to their metrics dicts
         (same structure used by `plot_training_metrics`).
   """
   if not model_metrics_dict:
      raise ValueError("model_metrics_dict must contain at least one model entry.")

   fig, axs = plt.subplots(1, 2, figsize=(16, 5))

   for model_name, metrics in model_metrics_dict.items():
      history = metrics.get("history")
      if history is None:
         raise ValueError(f"Missing 'history' for model '{model_name}'.")

      mae = history.get("mae")
      val_mae = history.get("val_mae")
      loss = history.get("loss")
      val_loss = history.get("val_loss")

      if mae:
         axs[0].plot(range(1, len(mae) + 1), mae, label=f"{model_name} Train")
      if val_mae:
         axs[0].plot(range(1, len(val_mae) + 1), val_mae, linestyle="--", label=f"{model_name} Val")

      if loss:
         axs[1].plot(range(1, len(loss) + 1), loss, label=f"{model_name} Train")
      if val_loss:
         axs[1].plot(range(1, len(val_loss) + 1), val_loss, linestyle="--", label=f"{model_name} Val")

   axs[0].set_title("MAE per Epoch (All Models)")
   axs[0].set_xlabel("Epoch")
   axs[0].set_ylabel("MAE")
   axs[0].grid(True, linestyle=":")
   axs[0].legend()

   axs[1].set_title("Loss per Epoch (All Models)")
   axs[1].set_xlabel("Epoch")
   axs[1].set_ylabel("Loss")
   axs[1].grid(True, linestyle=":")
   axs[1].legend()

   plt.tight_layout()
   plt.show()
