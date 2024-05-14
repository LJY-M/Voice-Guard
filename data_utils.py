import copy
import os
import pickle
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.signal import lfilter

from models import AdaInVC

from generate_masking_threshold import generate_th, generate_th


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0).cuda()
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 3, 'The number of dimensions of input tensor must be 3!'
        # reflect padding to match lengths of in/out
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter)


class InversePreEmphasis(torch.nn.Module):
    """
    Implement Inverse Pre-emphasis by using RNN to boost up inference speed.
    """

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.rnn = torch.nn.RNN(1, 1, 1, bias=False, batch_first=True)
        # use originally on that time
        self.rnn.weight_ih_l0.data.fill_(1)
        # multiply coefficient on previous output
        self.rnn.weight_hh_l0.data.fill_(self.coef)

    def forward(self, input: torch.tensor) -> torch.tensor:
        x, _ = self.rnn(input.transpose(1, 2))
        return x.transpose(1, 2)

def normalize_tensor(mel: torch.Tensor, attr: Dict) -> np.array:
    mean, std = attr["mean"], attr["std"]
    mel = torch.div(torch.sub(mel, torch.from_numpy(mean).cuda()), torch.from_numpy(std).cuda())
    return mel


def wav2mel_tensor(
    wav: torch.Tensor,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
    attr: Dict,
):

    preemp = PreEmphasis(coef=preemph)

    wav = wav.unsqueeze(0).unsqueeze(0)
    preemp_wav = preemp(wav).squeeze(0).squeeze(0)
    linear = torch.stft(input=preemp_wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        window=torch.hann_window(win_length).cuda(), center=True, pad_mode='reflect',
                        normalized=False, onesided=True, return_complex=False)
    mag = torch.sqrt(linear.pow(2).sum(-1) + (1e-9))

    mel_basis = torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)).cuda()
    mel = torch.matmul(mel_basis, mag)

    mel = torch.tensor([20]).cuda() * torch.log10(torch.maximum(torch.tensor([1e-5]).cuda(), mel))
    mel = torch.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mel = mel.T

    mel = normalize_tensor(mel, attr)

    return mel.T.unsqueeze(0).cuda()


def file2wav_mask(
    audio_path: str,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> Tuple[np.array, np.array, np.array]:
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    wav, _ = librosa.effects.trim(wav, top_db=top_db)
    wav = np.clip(wav, -1, 1)

    theta_xs, psd_max = generate_th(wav, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    return wav, theta_xs, psd_max