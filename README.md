An implementation of the "[Deep Content-User Embedding Model for Music Recommendation](https://arxiv.org/abs/1807.06786)" model in PyTorch.

---

### Description

This project uses [*Spotify*'s Million Playlist Dataset](https://research.atspotify.com/the-million-playlist-dataset-remastered/). It utilizes only 0.5% of its data which represents 5000 playlists and around 110,000 unique tracks. Additionally, [*Deezer*'s API](https://developers.deezer.com/api) is used to acquire the tracks' mel-spectrograms.

### Usage

    python run.py [-h] [--mpd FILE] [--mels FILE] [--nb-negatives NB] [--mel-length MEL_LENGTH] [--embedding-size NB] [--epochs NB] [--batch-size NB] {stats,fetch-mels,train}

**Positional arguments**:

- `{stats,fetch-mels,train}` display dataset related stats; fetch the audio files and generate the mel spectrograms; train the model

**Optional arguments**:

- `--mpd` FILE            Directory where the Million Playlist Dataset csv files are located.
- `--mels` FILE           Path where the mel-spectrograms are or will be extracted to.
- `--nb-negatives` NB     Size of the track negative sampling.
- `--mel-length` NB       Trims or loops audio signals to ensure all have the same length.
- `--embedding-size` NB   Embedding size of the playlist and tracks.
- `--epochs` NB
- `--batch-size` NB

---

### Requirements

```
# others
torch
numpy
tqdm
# required to fetch the audio data and audio processing
ffmpeg
librosa
```

### Disclaimer

The provided code is not designed to download and store audio tracks locally which would go against Deezer's policy. In this project, the 30 second audio samples provided by Deezer are momentarily cached in the file system before being removed. The conversion to mel-spectrogram is irreversible since it has a loss of information.
