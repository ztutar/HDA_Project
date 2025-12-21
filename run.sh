
pip install -e .

echo "▶️  Starting training for Base CNN model..."
python main.py --model base_cnn --config base.yaml

echo "▶️  Starting training for Skip-Connection CNN model..."
python main.py --model skipcon_cnn --config skipcon.yaml

echo "▶️  Starting training for Inception CNN model..."
python main.py --model inception_cnn --config inception.yaml

echo "✅  All models finished!"
