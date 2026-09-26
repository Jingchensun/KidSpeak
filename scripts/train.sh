#!/usr/bin/env bash
# Usage: bash scripts/train.sh [whisper_model] [num_gpus] [epochs]
# e.g.   bash scripts/train.sh small 4 10
set -e
WHISPER=${1:-small}
NUM_GPUS=${2:-4}
EPOCHS=${3:-10}

deepspeed --num_gpus ${NUM_GPUS} --master_port 28458 train.py \
  --train_file dataset/json/merged_train.json \
  --audio_root dataset \
  --whisper_model ${WHISPER} \
  --epochs ${EPOCHS} \
  --save_dir checkpoint/kidspeak_${WHISPER}
