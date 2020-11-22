import os
from argparse import ArgumentParser
from statistics import mean

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from helpers import bar
from fetching import fetch_mpd, fetch_mels
from processing import processing, split
from encoder import PlaylistEncoder, MelEncoder
from dataset import MPDSubset
from model import DCUE, DCUEWrapper

def main(args):
    playlists, tracks = fetch_mpd(args.mpd)

    if args.mode == 'stats':
        size = [len(p) for p in playlists]
        occurencies = [len(t[-1]) for t in tracks]
        print('%i playlists and %i unique tracks' % (len(playlists), len(tracks)))
        print('playlist sizes: min=%i ; avg=%i ; max=%i' % (min(size), mean(size), max(size)))
        bar(occurencies)

    elif args.mode == 'fetch-mels':
        tracks = [t for t in tracks if len(t[-1]) > 1] # remove tracks with a single occurence
        fails = fetch_mels(tracks, args.mels)
        print('failed to find and download %i tracks' % len(fails))

    elif args.mode == 'train':
        track_list = processing(tracks, args.mels)[:10000]
        train_tracks, valid_tracks, test_tracks = split(track_list, shuffle=False)

        playlist_enc = PlaylistEncoder(train_tracks, accumulate=True)
        mel_enc = MelEncoder(args.mels, args.mel_mean, args.mel_std, args.mel_length)

        train_set = MPDSubset(train_tracks, track_list, playlists, playlist_enc, mel_enc,
            nb_negatives=args.nb_negatives)
        train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

        valid_set = MPDSubset(valid_tracks, track_list, playlists, playlist_enc, mel_enc,
            nb_negatives=args.nb_negatives)
        valid_dl = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

        model = DCUE(len(playlist_enc), embedding_size=args.embedding_size)
        system = DCUEWrapper(model, train_dl, valid_dl, args.save)

        system.train(args.epochs, load=args.load)

parser = ArgumentParser()

parser.add_argument('mode', type=str, default='train', choices=['stats', 'analyse-mels', 'fetch-mels', 'train'])
parser.add_argument('--mpd', type=str, metavar='FILE', default='../datasets/MPD/subset',
    help='Directory where the Million Playlist Dataset csv files are located.')
parser.add_argument('--mels', type=str, metavar='FILE', default='../datasets/mels',
    help='Path where the mel-spectrograms are or will be extracted to.')
parser.add_argument('--load', type=str, metavar='FILE', default=None)
parser.add_argument('--save', type=str, metavar='FILE', default='out.pth')
parser.add_argument('--nb-negatives', type=int, metavar='NB', default=20,
    help='Size of the track negative sampling.')
parser.add_argument('--mel-length', metavar='NB', type=int, default=1323,
    help='Trims or loops audio signals to ensure all have the same length.')
parser.add_argument('--mel-mean', metavar='NB', type=float, default=-41.8637,
    help='Mean of the mel spectrogtrams in the training set.')
parser.add_argument('--mel-std', metavar='NB', type=float, default=14.2748,
    help='Standard deviation of the mel spectrogtrams in the training set.')
    parser.add_argument('--limit', metavar='NB', type=int, default=10000,
        help='Limits the number of playlists to load.')

parser.add_argument('--embedding-size', metavar='NB', type=int, default=128,
    help='Embedding size of the playlist and tracks.')
parser.add_argument('--epochs', metavar='NB', type=int, default=20)
parser.add_argument('--batch-size', metavar='NB', type=int, default=10)
args = parser.parse_args()

if os.path.isfile(args.save):
    print('file "%s" already exists, you risk losing it' % args.save)

main(args)
