import torch
import torch.nn as nn
import torchaudio
import whisper

SAMPLE_RATE = 16000


def _patch_whisper_layernorm():
    """Whisper's LayerNorm upcasts the input to fp32, which breaks when the encoder
    weights are in fp16/bf16. Cast the input to the weight dtype instead."""
    def forward(self, x):
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return nn.LayerNorm.forward(self, x)
    whisper.model.LayerNorm.forward = forward


_patch_whisper_layernorm()


def load_log_mel(audio_paths, n_mels, device):
    """Load a batch of audio files and return Whisper log-Mel spectrograms of shape [B, n_mels, 3000].

    Each clip is resampled to 16 kHz, flattened to 1-D and padded/trimmed to 30 s.
    An empty path yields an all-zero spectrogram (text-only sample).
    """
    mels = []
    for path in audio_paths:
        if path:
            waveform, sr = torchaudio.load(path, normalize=True)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            waveform = whisper.pad_or_trim(waveform.flatten())
            mel = whisper.log_mel_spectrogram(waveform, n_mels=n_mels)
        else:
            mel = torch.zeros(n_mels, 3000)
        mels.append(mel.to(device))
    return torch.stack(mels, dim=0)
