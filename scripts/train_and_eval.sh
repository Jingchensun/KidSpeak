#!/usr/bin/env bash
# Train for <epochs>, evaluate every epoch's checkpoint on the test split, then plot the curves.
# Usage: bash scripts/train_and_eval.sh [whisper_model] [num_gpus] [epochs]
# e.g.   bash scripts/train_and_eval.sh small 4 10
set -e
WHISPER=${1:-small}
NUM_GPUS=${2:-4}
EPOCHS=${3:-10}

bash scripts/train.sh ${WHISPER} ${NUM_GPUS} ${EPOCHS}
for ((E = 0; E < EPOCHS; E++)); do
  bash scripts/eval.sh ${WHISPER} ${E} test ${NUM_GPUS}
done
python scripts/plot_curves.py --run outputs/kidspeak_${WHISPER} --log checkpoint/kidspeak_${WHISPER}/train.log
