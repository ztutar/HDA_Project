
echo "Running FUSION model..."
python main.py --model fusion_cnn --config fusion.yaml

echo "Running FUSION model with Gender..."
python main.py --model fusion_cnn --config fusion_gender.yaml

echo "All models finished!"
