"""Run a trained KidSpeak checkpoint on a test split.

Every human turn of every test conversation is asked in order (multi-turn, with the model's own
previous answers as history). Results are written as JSON lines:
    {"audio": ..., "question": ..., "prediction": ..., "answer": ...}

Example:
    python inference.py --ckpt checkpoints/kidspeak_small/pytorch_model_9.pt \
        --test_file data/json/ultrasuite_disorder_test.json --audio_root data \
        --output results/kidspeak_small/ultrasuite_disorder.jsonl
"""
import argparse
import json
import os

import torch
import yaml
from tqdm import tqdm

from kidspeak import KidSpeak


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='pytorch_model_<epoch>.pt saved by train.py')
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--audio_root', type=str, default='data')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--config', type=str, default=None,
                        help='defaults to the config.yaml saved next to the checkpoint')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1.0)
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

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fout:
        for sample in tqdm(test_set):
            audio_path = os.path.join(args.audio_root, sample['audio_name'])
            audio_embeds = model.encode_audio([audio_path])
            conversation = sample['conversation']
            history = []
            for i in range(0, len(conversation), 2):
                question, answer = conversation[i]['value'], conversation[i + 1]['value']

                # Previous turns are prepended in the same format used during training
                prompt = ''
                for j, (q, a) in enumerate(history):
                    prompt += ('' if j == 0 else ' Human: ') + f'{q}\n### Assistant: {a}\n###'
                prompt += f' Human: {question}' if history else question

                prediction = model.generate(prompt, audio_embeds, max_new_tokens=args.max_new_tokens,
                                            top_p=args.top_p, temperature=args.temperature)
                history.append((question, prediction))
                fout.write(json.dumps({'audio': sample['audio_name'], 'question': question,
                                       'prediction': prediction, 'answer': answer}, ensure_ascii=False) + '\n')
            fout.flush()


if __name__ == '__main__':
    main()
