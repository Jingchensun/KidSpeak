"""Run a trained KidSpeak checkpoint on a test split.

Every human turn of every test conversation is asked in order (multi-turn, with the model's own
previous answers as history). Samples are processed in batches, one dialogue turn at a time.
Results are written as JSON lines:
    {"audio": ..., "question": ..., "prediction": ..., "answer": ...}

Example:
    python inference.py --ckpt checkpoint/kidspeak_small/pytorch_model_9.pt \
        --test_file dataset/json/ultrasuite_disorder_test.json --audio_root dataset \
        --output outputs/kidspeak_small/ultrasuite_disorder.jsonl

For multi-GPU evaluation, run one process per GPU with --num_shards N --shard_id i; each process
handles a contiguous chunk of the test set and writes <output>.shard<i>. scripts/eval.sh does this
and merges the shards in order.
"""
import argparse
import json
import math
import os

import torch
import yaml
from tqdm import tqdm

from kidspeak import KidSpeak


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='pytorch_model_<epoch>.pt saved by train.py')
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--audio_root', type=str, default='dataset')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--config', type=str, default=None,
                        help='defaults to the config.yaml saved next to the checkpoint')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_id', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = args.config or os.path.join(os.path.dirname(args.ckpt), 'config.yaml')
    with open(config) as f:
        cfg = yaml.safe_load(f)

    model = KidSpeak(**cfg)
    model.load_trainable_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model = model.eval().half().cuda()

    with open(args.test_file) as f:
        test_set = json.load(f)
    chunk = math.ceil(len(test_set) / args.num_shards)
    test_set = test_set[args.shard_id * chunk:(args.shard_id + 1) * chunk]
    output = args.output if args.num_shards == 1 else f'{args.output}.shard{args.shard_id}'

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as fout:
        for start in tqdm(range(0, len(test_set), args.batch_size)):
            batch = test_set[start:start + args.batch_size]
            audio_embeds = model.encode_audio([os.path.join(args.audio_root, s['audio_name']) for s in batch])
            histories = [[] for _ in batch]
            num_turns = max(len(s['conversation']) // 2 for s in batch)
            for t in range(num_turns):
                # samples that still have a question at turn t
                active = [i for i, s in enumerate(batch) if 2 * t < len(s['conversation'])]
                prompts = []
                for i in active:
                    question = batch[i]['conversation'][2 * t]['value']
                    # previous turns are prepended in the same format used during training
                    prompt = ''
                    for j, (q, a) in enumerate(histories[i]):
                        prompt += ('' if j == 0 else ' Human: ') + f'{q}\n### Assistant: {a}\n###'
                    prompt += f' Human: {question}' if histories[i] else question
                    prompts.append(prompt)
                predictions = model.generate(prompts, audio_embeds[active], max_new_tokens=args.max_new_tokens,
                                             top_p=args.top_p, temperature=args.temperature)
                for i, prediction in zip(active, predictions):
                    histories[i].append((batch[i]['conversation'][2 * t]['value'], prediction))

            for s, history in zip(batch, histories):
                for t, (question, prediction) in enumerate(history):
                    fout.write(json.dumps({'audio': s['audio_name'], 'question': question, 'prediction': prediction,
                                           'answer': s['conversation'][2 * t + 1]['value']}, ensure_ascii=False) + '\n')
            fout.flush()


if __name__ == '__main__':
    main()
