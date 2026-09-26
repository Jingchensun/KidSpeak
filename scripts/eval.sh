#!/usr/bin/env bash
# Usage: bash scripts/eval.sh [whisper_model] [epoch]
# Runs inference on the three test sets and computes the metrics.
set -e
WHISPER=${1:-small}
EPOCH=${2:-9}
CKPT=checkpoints/kidspeak_${WHISPER}/pytorch_model_${EPOCH}.pt
OUT=results/kidspeak_${WHISPER}/epoch_${EPOCH}

for TASK in ultrasuite_disorder talkbank_v1_3_enni_post english_children; do
  python inference.py \
    --ckpt ${CKPT} \
    --test_file data/json/${TASK}_test.json \
    --audio_root data \
    --output ${OUT}/${TASK}.jsonl
  python compute_metrics.py --pred ${OUT}/${TASK}.jsonl
done
