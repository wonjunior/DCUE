An implementation of the "[Deep Content-User Embedding Model for Music Recommendation](https://arxiv.org/abs/1807.06786)" model in PyTorch.

---

### Description

This project uses [*Spotify*'s Million Playlist Dataset](https://research.atspotify.com/the-million-playlist-dataset-remastered/). It utilizes only 0.5% of its data which represents 5000 playlists and around 110,000 unique tracks. Additionally, [*Deezer*'s API](https://developers.deezer.com/api) is used to acquire the tracks' mel-spectrograms.

### Usage

    python run.py [-h] [--epochs NB] [--batch-size NB] [--criterion {multi_margin,cross_entropy}] [--margin MARGIN] [--mpd FILE] [--mels FILE] [--load FILE] [--save FILE] [--nb-negatives NB] [--mel-length NB] [--embedding-size NB] [--mel-bins NB] [--mel-mean NB] [--mel-std NB] [--limit NB] [--mel-min NB] [--mel-max NB] {stats,analyse-mels,fetch-mels,train,show}

**Positional arguments**:

- `mode`, choose from:
    - `stats`: Display dataset related stats.
    - `fetch-mels`: Fetch the audio files and generate the mel spectrograms.
    - `analyse-mels`: Get mel spectrogram's value range and distribution.
    - `train`: Train the model.
    - `show`: Plot the training results.

**Optional arguments**:

- `--epochs NB`
- `--batch-size NB`
- `--criterion`           Choose from {multi_margin,cross_entropy}
- `--margin MARGIN`       Margin for multi-margin loss.
- `--mpd FILE`            Directory where the Million Playlist Dataset csv files are located.
- `--mels FILE`           Path where the mel-spectrograms are or will be extracted to.
- `--load FILE`           File loaded to resume training, finetune or plot results.
- `--save FILE`           Output file to save model's state.
- `--nb-negatives NB`     Size of the track negative sampling.
- `--mel-length NB`       Trims or loops audio signals to ensure all have the same length.
- `--embedding-size NB`   Embedding size of the playlist and tracks.
- `--mel-bins NB`         Number of bins in the mel spectrograms, i.e. number of channels in 1D-CNN.
- `--mel-mean NB`         Mean of the mel spectrogtrams in the training set.
- `--mel-std NB`          Standard deviation of the mel spectrogtrams in the training set.
- `--limit NB`            Limits the number of playlists to load.
- `--mel-min NB`          Minimum value of the mel-spectrogtram.
- `--mel-max NB`          Maximum value of the mel-spectrogtram.


---

### Requirements

```
# others
torch
numpy
tqdm
matplotlib
# required to fetch the audio data and audio processing
ffmpeg
librosa
```

### Disclaimer

The provided code is not designed to download and store audio tracks locally which would go against Deezer's policy. In this project, the 30 second audio samples provided by Deezer are momentarily cached in the file system before being removed. The conversion to mel-spectrogram is irreversible since it has a loss of information.
