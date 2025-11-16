#!/bin/bash

echo "Running ROI model..."
python main.py --model roi_cnn --config roi_only.yaml

echo "Running GLOBAL model..."
python main.py --model global_cnn --config global_only.yaml

echo "Running FUSION model..."
python main.py --model fusion_cnn --config fusion.yaml

echo "All models finished!"
