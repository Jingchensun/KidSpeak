#!/usr/bin/env bash
# Usage: bash scripts/eval.sh [whisper_model] [epoch] [split]
# Runs inference on the three datasets of a split (test | val) and computes the metrics.
set -e
WHISPER=${1:-small}
EPOCH=${2:-9}
SPLIT=${3:-test}
CKPT=checkpoint/kidspeak_${WHISPER}/pytorch_model_${EPOCH}.pt
OUT=outputs/kidspeak_${WHISPER}/epoch_${EPOCH}/${SPLIT}

for TASK in ultrasuite_disorder talkbank_v1_3_enni_post english_children; do
  python inference.py \
    --ckpt ${CKPT} \
    --test_file dataset/json/${TASK}_${SPLIT}.json \
    --audio_root dataset \
    --output ${OUT}/${TASK}.jsonl
  python compute_metrics.py --pred ${OUT}/${TASK}.jsonl
done
