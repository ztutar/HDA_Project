#!/usr/bin/env bash
set -euo pipefail

# Run the InSkipCon model across all combinations of CLAHE/augment/use_gender flags.

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

IMAGE_SIZE=256
BATCH_SIZE=8
BASE_FILTERS=32
NUM_A=2
NUM_B=3
NUM_C=1
SCALE_A=0.17
SCALE_B=0.1
SCALE_C=0.2
DENSE_UNITS=128
DROPOUT=0.2
EPOCHS=70
PATIENCE=10
LR=0.001
RESULTS_CSV="experiments/train_results_summary.csv"
PERFORM_TEST=true

for clahe in true false; do
for augment in true false; do
for use_gender in true false; do
   cfg_path="$TMP_DIR/inskipcon_c${clahe}_a${augment}_g${use_gender}.yaml"
   cat > "$cfg_path" <<EOF
data:
  image_size: ${IMAGE_SIZE}
  clahe: ${clahe}
  augment: ${augment}
  batch_size: ${BATCH_SIZE}

model:
  base_filters: ${BASE_FILTERS}
  num_a_blocks: ${NUM_A}
  num_b_blocks: ${NUM_B}
  num_c_blocks: ${NUM_C}
  scale_a: ${SCALE_A}
  scale_b: ${SCALE_B}
  scale_c: ${SCALE_C}
  dense_units: ${DENSE_UNITS}
  dropout_rate: ${DROPOUT}
  use_gender: ${use_gender}

training:
  epochs: ${EPOCHS}
  patience: ${PATIENCE}
  learning_rate: ${LR}
  results_csv: "${RESULTS_CSV}"
  perform_test: ${PERFORM_TEST}
EOF

   echo "▶️  Running InSkipCon: clahe=${clahe}, augment=${augment}, use_gender=${use_gender}"
   python main.py --model inskipcon --config "$cfg_path"
done
done
done

echo "✅  InSkipCon sweep complete."
