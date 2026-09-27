#!/usr/bin/env bash
# Usage: bash scripts/eval.sh [whisper_model] [epoch | untrained] [split] [num_gpus]
# e.g.   bash scripts/eval.sh small 9 test 4
#        bash scripts/eval.sh small untrained test 4   # model before instruction tuning
# For each dataset of the split (default: test), the set is split into <num_gpus> contiguous shards that
# run in parallel (one per GPU), the shards are merged in order, and the metrics are computed.
set -e
WHISPER=${1:-small}
EPOCH=${2:-9}
SPLIT=${3:-test}
NUM_GPUS=${4:-4}
if [ "${EPOCH}" = "untrained" ]; then
  # the untrained model mostly emits nothing but does not stop; 64 new tokens is enough to score it
  MODEL_ARGS="--config configs/kidspeak.yaml --max_new_tokens 64"
  OUT=outputs/kidspeak_${WHISPER}/untrained/${SPLIT}
else
  MODEL_ARGS="--ckpt checkpoint/kidspeak_${WHISPER}/pytorch_model_${EPOCH}.pt"
  OUT=outputs/kidspeak_${WHISPER}/epoch_${EPOCH}/${SPLIT}
fi
mkdir -p ${OUT}

for TASK in ultrasuite enni english_children; do
  PRED=${OUT}/${TASK}.jsonl
  PIDS=()
  for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=${i} python inference.py \
      ${MODEL_ARGS} \
      --test_file dataset/json/${TASK}_${SPLIT}.json \
      --audio_root dataset \
      --output ${PRED} \
      --num_shards ${NUM_GPUS} --shard_id ${i} > ${PRED}.shard${i}.log 2>&1 &
    PIDS+=($!)
  done
  for PID in "${PIDS[@]}"; do
    wait ${PID} || { echo "inference failed, see ${PRED}.shard*.log"; exit 1; }
  done
  cat $(for ((i = 0; i < NUM_GPUS; i++)); do echo ${PRED}.shard${i}; done) > ${PRED}
  rm -f ${PRED}.shard*
  python compute_metrics.py --pred ${PRED}
done
