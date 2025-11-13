# Bone Age Prediction - HDA Project

Training and experimentation toolkit for the RSNA pediatric hand X-ray dataset. The scope is to predict bone age from hand X-rays with minimal error while experimenting with alternative preprocessing pipelines (CLAHE, augmentation) and model designs (Global, ROI, Fusion; with/without gender) to understand which combinations generalise best.

## Highlights

- **Multiple model families**: `GlobalCNN`, `ROI_CNN`, and `Fusion_CNN` trainers share utilities but can be run independently.
- **Automatic ROI pipeline**: carpal/metacarpus & phalanx crops are created with the locator in `src/BAP/roi` and cached under `data/cropped_rois`.
- **Config-driven experiments**: YAML files in `experiments/configs` describe data, model, ROI, and training settings.
- **Experiment tracking**: checkpoints land in `experiments/checkpoints`, summaries append to `experiments/train_results_summary.csv`, and curated exports live in `model_checkpoint/`.

## Repository Layout

```text
HDA_Project/
├── main.py                     # Entry point that wires configs, datasets, and trainers
├── data/
│   ├── raw/                    # Optionally store raw RSNA images (train/validation/test)
│   ├── cropped_rois/           # Auto-generated ROI crops + heatmaps
│   └── metadata/               # CSV splits consumed by the tf.data pipelines
├── experiments/
│   ├── configs/                # global_only.yaml, roi_only.yaml, fusion.yaml, …
│   ├── checkpoints/            # stores .keras weights, TB logs, and .log files
│   └── train_results_summary.csv
├── src/BAP/
│   ├── models/                 # Fusion_CNN.py, Global_CNN.py, ROI_CNN.py
│   ├── roi/                    # ROI locator/extractor built on Grad-CAM peaks
│   ├── training/               # Trainer scripts, callbacks, summaries
│   ├── utils/                  # Config loader, dataset utilities, seed & path helpers
│   └── visualization/          # gradcam.py, overlay.py, plots.py
├── BoneAgePrediction.ipynb     # Interactive notebook for exploratory work
├── model_checkpoint/           # Keras exports + metrics/results dicts from notebook
├── report/                     # Slides and report
├── pyproject.toml              # Package metadata and entry point required for env setup
└── README.md                   # You are here!
```

## Getting Started

### Prerequisites

- Python ≥ 3.9 (3.11+ recommended) and pip.
- CUDA-capable GPU for reasonable training times; Linux is strongly recommended because TensorFlow ≥2.11 drops native Windows CUDA builds (per the TF warning: on Windows you must use WSL2 or fall back to `tensorflow-cpu` + DirectML).

### Environment setup

```bash
git clone https://github.com/ztutar/HDA_Project.git
cd HDA_Project
python -m venv .venv
source .venv/bin/activate           
pip install --upgrade pip
pip install -e .                    # installs base deps 
# If you have CUDA support, also install:
pip install tensorflow[and-cuda]
```

If the environment already ships with TensorFlow (e.g., Google Colab’s GPU runtimes), you can skip the extra install or run `pip install -e . --no-deps` to avoid replacing the preloaded build.

### Dataset & metadata

1. The first call to `get_rsna_dataset()` (triggered automatically from `main.py`) uses `kagglehub` to download `ipythonx/rsna-bone-age` into your Kaggle cache.
2. CSV metadata in `data/metadata/{train,validation,test}.csv` should provide at least `Image ID`, `Bone Age (months)`, and `male` columns aligned with the downloaded images.
3. Alternatively, download the RSNA Bone Age Dataset from [Stanford’s Box mirror](https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739) if you prefer to keep datasets inside the repo instead of the Kaggle cache. But this requires manual update of the data paths in `main.py`.

## Notebooks & analysis

<a href="https://colab.research.google.com/github/ztutar/HDA_Project/blob/main/BoneAgePrediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- `BoneAgePrediction.ipynb` – Interactive pipeline walkthrough, rapid prototyping, and qualitative assessment.
- `model_checkpoint/` – `.keras` exports and model_metrics/model_results dictionaries saved from the notebook.

## Training entry points

Run trainings via `main.py`, which normalises model aliases and handles dataset paths, seed setting, and incremental save directories:

```bash
python main.py --model fusion --config fusion.yaml
```

| `--model` flag | Trainer (file)                 | Description                                                 | Default config |
|----------------|--------------------------------|-------------------------------------------------------------|----------------|
| `global`, `global_cnn` | `BAP.training.train_GlobalCNN` | CNN on full-resolution hands; CLAHE/augmentation + optional gender input. | `experiments/configs/global_only.yaml` |
| `roi`, `roi_cnn`       | `BAP.training.train_ROI_CNN`   | Two-branch regressor on carpal & metacarpus/phalanx crops; CLAHE/augmentation + optional gender input. | `experiments/configs/roi_only.yaml`    |
| `fusion`, `fusion_cnn` | `BAP.training.train_Fusion_CNN`| Fuses global stream with ROI embeddings; CLAHE/augmentation + optional gender input.| `experiments/configs/fusion.yaml`      |

If `--config` is omitted, defaults defined in `BAP.utils.config` are used. Each run saves under `experiments/checkpoints/<Model>/<config_name>_<run_id>/` where callbacks store the best `.keras` weights, TensorBoard logs, copied config, and history CSVs.

### ROI lifecycle

- The ROI & Fusion trainers look under `data/cropped_rois/<split>/{carpal,metaph,heatmaps}`.
- Missing crops trigger `BAP.roi.ROI_locator.train_locator_and_save_rois`, which leverages Grad-CAM peaks from a pretrained `GlobalCNN` checkpoint (`roi.locator.pretrained_model_path`) to cut and persist crops.
- Once created, crops are reused, which keeps subsequent experiments fast.

## Experiments & outputs

- `experiments/checkpoints/` – Per-run folders with weights (`*.keras`), TensorBoard logs, history CSVs, and metadata about the run.
- `experiments/train_results_summary.csv` – Aggregates each run’s hyperparameters plus train/val/test metrics, parameter counts, and timing.
- `data/cropped_rois/` – Cached ROI crops and optional heatmaps grouped by split and ROI type.

## Configuration reference

Configurations are hierarchical; all sections are optional and validated before use. Example:

```yaml
data:
  image_size: 256
  clahe: true
  augment: false
  batch_size: 16

roi:
  locator:
    roi_path: "data/cropped_rois"
    pretrained_model_path: "model_checkpoint/GlobalCNN_best.keras"
  extractor:
    roi_size: 128
    heatmap_threshold: 0.25
    save_heatmaps: true

model:
  global_channels: [32, 64, 128]
  roi_channels: [32, 64]
  fusion_dense_units: [256, 128]
  dropout_rate: 0.2
  use_gender: true

training:
  epochs: 50
  patience: 10
  learning_rate: 3e-4
  results_csv: "experiments/train_results_summary.csv"
  perform_test: true
```

Use these knobs to control augmentation, CLAHE, ROI crop sizes, channel widths, dropout, patience, and logging destinations. Any unknown keys are ignored.
