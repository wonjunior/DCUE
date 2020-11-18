import librosa
import numpy as np
import matplotlib.pyplot as plt

class MelSpectrogram():
    def __init__(self, sr, n_fft, win_length, hop_length, n_mels):
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, fp):
        y, sr = librosa.load(fp, sr=self.sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB.astype(np.float32)

    def show(self, fp):
        plt.figure(figsize=(10,8))
        mel = np.load(fp)
        plt.imshow(mel)
