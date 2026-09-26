import json
import os

from torch.utils.data import Dataset


class AudioInstructionDataset(Dataset):
    """Audio instruction-tuning data.

    Each item of the JSON file looks like:
        {"audio_name": "KIDS/.../xxx.wav",
         "conversation": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]}
    `audio_name` is resolved relative to `audio_root`.
    """

    def __init__(self, data_path, audio_root):
        with open(data_path) as f:
            data = json.load(f)
        self.audio_paths = [os.path.join(audio_root, item['audio_name']) for item in data]
        self.conversations = [item['conversation'] for item in data]
        print(f'[!] Loaded {len(self.audio_paths)} samples from {data_path}')

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, i):
        return self.audio_paths[i], self.conversations[i]

    @staticmethod
    def collate(instances):
        audio_paths, conversations = zip(*instances)
        return {'audio_paths': list(audio_paths), 'output_texts': list(conversations)}
