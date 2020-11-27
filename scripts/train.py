from torch.utils.data import DataLoader
from fetching import fetch_mpd
from processing import processing, split
from encoder import PlaylistEncoder, MelEncoder
from dataset import MPDSubset
from model import DCUE, DCUEWrapper

def train(args):
    playlists, tracks = fetch_mpd(args.mpd, args.limit)
    all_tracks = processing(tracks, args.mels)
    train_tracks, valid_tracks, test_tracks = split(all_tracks)

    mel_enc = MelEncoder(args.mels, mean=args.mel_mean, std=args.mel_std,
        length=args.mel_length, min=args.mel_min, max=args.mel_max)

    playlist_train_enc = PlaylistEncoder(train_tracks)
    train_set = MPDSubset(train_tracks, all_tracks, playlists, playlist_train_enc, mel_enc,
        nb_negatives=args.nb_negatives)
    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    playlist_valid_enc = PlaylistEncoder(valid_tracks)
    valid_set = MPDSubset(valid_tracks, all_tracks, playlists, playlist_valid_enc, mel_enc,
        nb_negatives=args.nb_negatives)
    valid_dl = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    model = DCUE(len(playlists), cnn_channels=args.mel_bins, emb_size=args.embedding_size)
    system = DCUEWrapper(model, train_dl, valid_dl, save=args.save, criterion=args.criterion,
        margin=args.margin, lr=args.learning_rate)

    system.fit(args.epochs, load=args.load)
