# HDA_Project

bone_age_project/
│
├── data/
│   ├── raw/
│   │   ├── train/                     # Original RSNA X-ray images
│   │   ├── valid/
│   │   └── test/
│   ├── processed/
│   │   ├── with_clahe/               # Optional preprocessed images (CLAHE applied)
│   │   └── cropped_rois/             # CAM-based ROI crops (carpal / phalange)
│   ├── metadata/
│   │   ├── boneage_train.csv         # age (months), gender, image_id
│   │   ├── splits.json               # train/val/test indices with seed
│   │   └── roi_coords.csv            # if you store ROI coordinates
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py         # TF Dataset pipeline, augmentations, preprocessing
│   │   ├── preprocess.py             # CLAHE, normalization, ROI extraction
│   │   └── split_data.py             # Train/val/test split creation
│   │
│   ├── models/
│   │   ├── base_blocks.py            # Basic conv blocks, attention (CBAM-lite)
│   │   ├── global_cnn.py             # Global-only CNN
│   │   ├── roi_cnn.py                # ROI-only CNNs
│   │   ├── fusion_cnn.py             # Global + ROI fusion CNN
│   │   └── model_factory.py          # Chooses model based on config
│   │
│   ├── training/
│   │   ├── train.py                  # Main training loop
│   │   ├── losses.py                 # Huber/MAE setup
│   │   ├── metrics.py                # MAE, RMSE, complexity (params, GMACs)
│   │   ├── callbacks.py              # Early stopping, LR schedulers, logging
│   │   └── evaluate.py               # Evaluate on test, compute metrics
│   │
│   ├── visualization/
│   │   ├── gradcam.py                # Generate CAMs for ROI extraction & explainability
│   │   ├── plots.py                  # Loss curves, age scatter plots, CAM overlays
│   │   └── report_figures.py         # Automated figure generation for paper/slides
│   │
│   ├── utils/
│   │   ├── config.py                 # YAML/JSON-based configuration parser
│   │   ├── logger.py                 # Experiment logging (e.g., TensorBoard, CSV)
│   │   └── seed_everything.py        # Fix random seeds for reproducibility
│   │
│   └── main.py                       # Entry point: loads config → builds → trains → evaluates
│
├── experiments/                      # Centralized experiments area (configs, logs, weights, results)
│   ├── logs/                         # Per-run logs (TensorBoard/CSV), timings, and copied config
│   ├── configs/
│   │   ├── global_only.yaml          # Global-only CNN on full hand X-rays
│   │   ├── roi_only.yaml             # ROI-only CNN on cropped carpal/phalange regions
│   │   ├── fusion.yaml               # Two-branch fusion model (global + ROI features)
│   │   └── clahe.yaml                # Ablation to measure CLAHE preprocessing impact
│   ├── checkpoints/
│   │   ├── global_only_best.h5       # Best global-only weights by val MAE
│   │   ├── roi_only_best.h5          # Best ROI-only weights by val MAE
│   │   └── fusion_best.h5            # Best fusion-model weights by val MAE
│   └── results_summary.csv           # Aggregated results: MAE/RMSE, params, GMACs, time, seed

│
├── reports/
│   ├── figures/                      # GradCAM visualizations, plots, etc.
│   ├── tables/                       # CSV or LaTeX tables for results
│   ├── presentation_slides.pptx
│   ├── project_report.tex
│   └── project_report.pdf
│
├── environment.yml                   # Conda env (Python 3.11, TensorFlow, OpenCV, etc.)
├── requirements.txt                  # For pip
├── README.md                         # Project overview, setup, instructions
└── LICENSE
