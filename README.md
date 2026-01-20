# Bone Age Prediction - ML4HD Project

```text
Name:   Zeynep TUTAR
ID:     2106038
```

Training and experimentation toolkit for the RSNA pediatric hand X-ray dataset. The scope is to predict bone age from hand X-rays with minimal error while experimenting with CNN variants (Base, SkipCon, Inception, InSkipCon) and config-driven training setups.

## Highlights

- **Four model families**: `BaseCNN`, `SkipCon_CNN`, `Inception_CNN`, and `InSkipCon_CNN` models that can be trained independently.
- **Config-driven experiments**: YAML files in `experiments/configs` describe data, model, and training settings.
- **Experiment tracking**: checkpoints land in `experiments/checkpoints`, summaries append to `experiments/train_results_summary.csv`, and curated exports live in `model_checkpoint/`.

## Repository Layout

```bash
Bone_Age_Prediction/
├── main.py                         # Entry point that wires configs, datasets, and trainers
├── data/
│   ├── metadata/                   # CSV splits consumed by the tf.data pipelines
│   └── img/                        # Optional local copies of RSNA images
├── experiments/
│   ├── configs/                    # YAML config files for experiments
│   ├── checkpoints/                # Stores .keras weights, TB logs, and .log files
│   └── train_results_summary.csv   # Aggregated run metrics + config values
├── src/BAP/
│   ├── models/                     # Base_CNN.py, SkipCon_CNN.py, Inception_CNN.py, InSkipCon_CNN.py
│   ├── training/                   # Trainer scripts, callbacks
│   └── utils/                      # Config loader, dataset utilities, plotting, summaries etc.
├── BoneAgePrediction.ipynb         # Interactive notebook for exploratory work
├── model_checkpoint/               # Keras exports + metrics/results dicts from notebook
├── report/                         # Slides and report
├── pyproject.toml                  # Package metadata and entry point required for env setup
└── README.md                       # You are here!
```

## Getting Started

### Prerequisites

- Python ≥ 3.9 (3.11+ recommended) and pip.
- CUDA-capable GPU for reasonable training times; Linux is strongly recommended because TensorFlow ≥2.11 drops native Windows CUDA builds (per the TF warning: on Windows you must use WSL2 or fall back to `tensorflow-cpu` + DirectML).

### Environment setup

```bash
# Clone the repository locally
git clone https://github.com/ztutar/HDA_Project.git

# Change into the project directory
cd HDA_Project

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .                    

# If you have CUDA support, also install:
pip install tensorflow[and-cuda]
```

If the environment already ships with TensorFlow (e.g., Google Colab GPU runtimes), you can skip the extra install or run `pip install -e . --no-deps` to avoid replacing the preloaded build.

### Dataset & metadata

1. The first call to `get_rsna_dataset()` (triggered automatically from `main.py`) uses `kagglehub` to download `ipythonx/rsna-bone-age` into your Kaggle cache.
2. CSV metadata in `data/metadata/{train,validation,test}.csv` must include `Image ID`, `Bone Age (months)`, and `male` columns aligned with the downloaded images (image IDs should match the `.png` filenames without extension).
3. Alternatively, download the RSNA Bone Age Dataset from [Stanford’s Box mirror](https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739) if you prefer to keep datasets inside the repo instead of the Kaggle cache. Update the data paths in `main.py` accordingly.

## Notebooks & analysis

<a href="https://colab.research.google.com/github/ztutar/HDA_Project/blob/main/BoneAgePrediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- `BoneAgePrediction.ipynb` – Interactive pipeline walkthrough, rapid prototyping, and qualitative assessment.
- `model_checkpoint/` – `.keras` exports and model_metrics/model_results dictionaries saved from the notebook.

## Training entry points

Run trainings via `main.py`, which normalises model aliases and handles dataset paths, seed setting, and incremental save directories:

```bash
python main.py --model base --config base.yaml
```

| `--model` flag | Trainer (file)                      | Description                                                                   | Default config |
|----------------|-------------------------------------|-------------------------------------------------------------------------------|----------------|
| `base`, `base_cnn`     | `BAP.training.train_BaseCNN`       | VGG-style Baseline CNN on hand radiographs; CLAHE/augmentation/gender options.   | `experiments/configs/base.yaml` |
| `skipcon`, `skipcon_cnn` | `BAP.training.train_SkipCon_CNN`  | ResNet-style blocks with skip connections; CLAHE/augmentation/gender options.      | `experiments/configs/skipcon.yaml` |
| `inception`, `inception_cnn` | `BAP.training.train_Inception_CNN` | Inception-V4 style stem and multi-branch blocks; CLAHE/augmentation/gender options. | `experiments/configs/inception.yaml` |
| `inskipcon`, `inskipcon_cnn` | `BAP.training.train_InSkipCon_CNN` | Inception-ResNet-style hybrid with residual scaling (InSkipCon); CLAHE/augmentation/gender options. | `experiments/configs/inskipcon.yaml` |

If `--config` is omitted, defaults defined in `BAP.utils.config` are used. Each run saves under `experiments/checkpoints/<Model>/<config_name>/<Model>_<config_name>_<run_id>/` where callbacks store the best `.keras` weights, TensorBoard logs, and history CSVs.

## Experiments & outputs

- `experiments/checkpoints/` – Per-run folders with weights (`*.keras`), TensorBoard logs, history CSVs, and metadata about the run.
- `experiments/train_results_summary.csv` – Aggregates each run’s hyperparameters plus train/val/test metrics, parameter counts, and timing.
- `model_checkpoint/` – Notebook exports for quick inspection and sharing.

## Configuration reference

Configurations are hierarchical; all sections are optional and validated before use. Example:

```yaml
data:
  image_size: 512
  clahe: true
  augment: false
  batch_size: 16

model:
  channels: [32, 64, 128]        # Used by BaseCNN
  dense_units: 128
  base_filters: 32               # Used by SkipCon_CNN, Inception_CNN, InSkipCon_CNN
  block_filters: [32, 64, 128, 256]
  blocks_per_stage: [2, 2, 2, 2]
  num_a_blocks: 2
  num_b_blocks: 3
  num_c_blocks: 1
  scale_a: 0.17
  scale_b: 0.1
  scale_c: 0.2
  use_gender: false
  dropout_rate: 0.2

training:
  epochs: 30
  patience: 10
  learning_rate: 3e-4
  results_csv: "experiments/train_results_summary.csv"
  perform_test: false
```

Adjust these knobs to control image preprocessing, channel widths, block layouts, dropout, and logging destinations. Unknown keys are ignored.
