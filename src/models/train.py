import torch
import torchaudio
import torch.nn    as nn
import torch.nn.functional as f
import torchaudio.functional as F
import torchaudio.transforms as T

from datasets import Dataset 
from torch.utils.data import DataLoader

import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio, display
from torchaudio.utils import download_asset
import librosa

from microphone import MicrophoneModel

TARGET_WAV_path = "../../data/s1_cut.wav"
SOURCE_WAV_path = "../../data/s1_iphone_cut.wav"
SAMPLE_RIR_PATH = "../../data/rir.wav"
# metadata = torchaudio.info(SAMPLE_WAV)
# print(metadata)
hparams = {
    'win_length' : 1024,
    'n_fft'      : 1024,
    'hop_length' : 256,
    'lr'         : 1e-4,
    'beta1'      : 0.5,
    'beta2'      : 0.9,
}
model = MicrophoneModel(hparams)

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
     # ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_rir_sample(*, resample=None, processed=False):
  rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
  if not processed:
    return rir_raw, sample_rate
  rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
  rir = rir / torch.norm(rir, p=2)
  rir = torch.flip(rir, [1])
  return rir, sample_rate

waveform_S, sample_rate = _get_sample(SOURCE_WAV_path, resample=16000)
waveform_T, sample_rate = _get_sample(TARGET_WAV_path, resample=None)

waveform_R, _ = get_rir_sample(resample=16000)
rir = waveform_R[:, int(sample_rate*1.01):int(sample_rate*1.3)]
rir = rir / torch.norm(rir, p=2)
rir = torch.flip(rir, [1])

print(waveform_S.shape, waveform_T.shape)
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)

#melspec = mel_spectrogram(waveform)

#ds = waveform_S.with_format("torch")
dataloader = DataLoader(waveform_S, batch_size=4)
for batch in dataloader:
     print(batch)
print(len(dataloader))

loss_fn = nn.L1Loss()
optimizer = model.get_optimizer()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #model.train()
    for batch, X_source in enumerate(dataloader):

        # Compute prediction error
        y = model(X_source, rir)
        y_mel = mel_spectrogram(y)
        x_mel = mel_spectrogram(waveform_T)
        loss = loss_fn(y_mel,x_mel)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X_source)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


epochs = 10000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
print("Done!")

#save model
torch.save(model.state_dict(), "./model.pth")
model = MicrophoneModel(hparams)
model.load_state_dict(torch.load("./model.pth"))

#inferance
y_aug = model(waveform_S,rir)
torchaudio.save('s1_aug.wav',y_aug, 16000, encoding="PCM_S", bits_per_sample=16)

