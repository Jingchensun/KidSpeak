"""Instruction-tune KidSpeak (LoRA + audio projection) with DeepSpeed.

Example (4 GPUs):
    deepspeed --num_gpus 4 train.py \
        --train_file data/json/merged_train.json --audio_root data \
        --save_dir checkpoints/kidspeak_small --whisper_model small
"""
import argparse
import json
import logging
import os
import random

import deepspeed
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from kidspeak import AudioInstructionDataset, KidSpeak


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--audio_root', type=str, default='data', help='directory that audio_name paths are relative to')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/kidspeak.yaml')
    parser.add_argument('--ds_config', type=str, default='configs/ds_config.json')
    parser.add_argument('--whisper_model', type=str, default=None, help='overrides whisper_model in --config')
    parser.add_argument('--local_rank', type=int, default=0)  # set by the deepspeed launcher
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.whisper_model:
        cfg['whisper_model'] = args.whisper_model
    return args, cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args, cfg = parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    set_seed(cfg['seed'])

    os.makedirs(args.save_dir, exist_ok=True)
    if rank == 0:
        logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'), level=logging.INFO,
                            format='%(asctime)s %(message)s')
        with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f)

    with open(args.ds_config) as f:
        ds_config = json.load(f)
    micro_bs = ds_config['train_micro_batch_size_per_gpu']

    # Data
    dataset = AudioInstructionDataset(args.train_file, args.audio_root)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg['seed'])
    loader = DataLoader(dataset, batch_size=micro_bs, sampler=sampler, drop_last=True,
                        num_workers=1, collate_fn=AudioInstructionDataset.collate, pin_memory=True)

    # Warmup-decay schedule over the whole run
    total_steps = cfg['epochs'] * len(dataset) // ds_config['train_batch_size']
    ds_config['scheduler']['params']['total_num_steps'] = total_steps
    ds_config['scheduler']['params']['warmup_num_steps'] = max(10, int(total_steps * cfg['warmup_rate']))

    # Model
    model = KidSpeak(**cfg)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config_params=ds_config,
    )

    pbar = tqdm(total=cfg['epochs'] * len(loader), disable=rank != 0)
    step = 0
    for epoch in range(cfg['epochs']):
        sampler.set_epoch(epoch)
        engine.train()
        for batch in loader:
            loss, token_acc = engine(batch)
            engine.backward(loss)
            engine.step()

            pbar.set_description(f'epoch {epoch} | loss {loss.item():.4f} | token_acc {token_acc * 100:.2f}')
            pbar.update(1)
            if rank == 0 and step % cfg['logging_step'] == 0:
                logging.info(f'epoch {epoch} step {step} loss {loss.item():.4f} token_acc {token_acc * 100:.2f}')
            step += 1

        # Only the LoRA adapters and the audio projection are saved (~tens of MB per epoch)
        torch.distributed.barrier()
        if rank == 0:
            path = os.path.join(args.save_dir, f'pytorch_model_{epoch}.pt')
            torch.save(model.trainable_state_dict(), path)
            print(f'[!] Saved {path}')


if __name__ == '__main__':
    main()
