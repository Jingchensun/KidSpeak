#!/usr/bin/env bash
# Usage: bash scripts/train.sh [whisper_model] [num_gpus]
# e.g.   bash scripts/train.sh small 4
set -e
WHISPER=${1:-small}
NUM_GPUS=${2:-4}

deepspeed --num_gpus ${NUM_GPUS} --master_port 28458 train.py \
  --train_file data/json/merged_train.json \
  --audio_root data \
  --whisper_model ${WHISPER} \
  --save_dir checkpoints/kidspeak_${WHISPER}
