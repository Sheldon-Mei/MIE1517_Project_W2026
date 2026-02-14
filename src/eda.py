from typing import List
from pathlib import Path

import random
import numpy as np
import librosa as lr


def to_spectrogram(
    input_signal: np.ndarray,
    n_fft=512, 
    hop_length=128
) -> np.ndarray:
    S = lr.stft(input_signal, n_fft=n_fft, hop_length=hop_length, center=False)
    S_db = lr.amplitude_to_db(S, ref=np.max)
    return S_db # [1 + n_fft/2, T]


# Pad the spectrogram
def pad_spectrograms(S_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Pads a list of spectrograms to the same number of frames.
    Works for a single batch; does not require knowing the max length in the dataset.
    """
    # Find max mel bins and max frames
    max_mels = max(s.shape[0] for s in S_list)
    max_frames = max(s.shape[1] for s in S_list)

    padded_S = []
    for S in S_list:

        # Pad mel bins (frequency axis)
        if S.shape[0] < max_mels:
            pad_mel = np.zeros((max_mels - S.shape[0], S.shape[1]))
            S = np.concatenate([S, pad_mel], axis=0)

        # Pad frames (time axis)
        if S.shape[1] < max_frames:
            pad_frame = np.zeros((S.shape[0], max_frames - S.shape[1]))
            S = np.concatenate([S, pad_frame], axis=1)

        # print(f"Padded spectrogram shape: {S.shape}")
        padded_S.append(S)

    return padded_S


# 3D plot of STFT
import plotly.graph_objects as go
 
samples = list((Path(__file__).parent / "data" / "raw").glob("*.wav"))
signal, sr = lr.load(samples[0], sr=16000)

# Simulated signal
sr = 16000
t = np.linspace(0, 1, sr)

# FFT
fft_freqs = np.fft.rfftfreq(len(signal), 1/sr)
fft_mag = np.abs(np.fft.rfft(signal))

# STFT Librosa
n_fft=2048
hop_length=512
Zxx = lr.stft(signal, n_fft=n_fft, hop_length=hop_length)
Zmag = np.abs(Zxx)
# Zmag = lr.amplitude_to_db(Zmag, ref=np.max)
f = lr.fft_frequencies(sr=sr, n_fft=n_fft)
tt = lr.frames_to_time(np.arange(Zmag.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)

# 3D surface plot
fig = go.Figure(data=[go.Surface(
    x=tt, 
    y=f, 
    z=Zmag, 
    colorscale='Viridis'
)])
fig.update_layout(scene=dict(
    xaxis_title='Time (s)',
    yaxis_title='Frequency (Hz)',
    zaxis_title='Magnitude'
))
# fig.show()
