import random

import numpy as np
from torch.utils.data import Dataset

class MPDSubset(Dataset): # using tid_to_index, tracks, nb_negatives
    def __init__(self, tracks, all_tracks, playlists, playlist_enc, mel_enc, nb_negatives=20):
        super(MPDSubset).__init__()
        self.tracks = tracks
        self.all_tracks = all_tracks
        self.playlists = playlists
        self.playlist_enc = playlist_enc
        self.mel_enc = mel_enc
        self.nb_negatives = nb_negatives

    def __getitem__(self, index):
        tid, artist, track, playlist_indices = self.tracks[index]
        pid = random.choice(playlist_indices)

        # Construct playlist-side input.
        playlist_encoded = self.playlist_enc(self.playlists[pid])

        # Construct positive item.
        pos_track = self.mel_enc(tid)

        # Construct negative sampling items.
        not_listened = np.squeeze(np.argwhere(playlist_encoded == 0), axis=1)
        neg_track_indices = np.random.choice(not_listened, self.nb_negatives, replace=False)
        neg_tracks = list(map(lambda i: self.mel_enc(self.all_tracks[i][0]), neg_track_indices))

        return pid, pos_track, neg_tracks

    def __len__(self):
        return len(self.tracks)
