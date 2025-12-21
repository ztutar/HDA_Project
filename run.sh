
pip install -e .

python main.py --model inception --config inception.yaml

python main.py --model densenet --config densenet.yaml

python main.py --model skipcon --config skipcon.yaml

echo "All models finished!"
