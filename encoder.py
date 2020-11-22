import os
import numpy as np

class PlaylistEncoder():
    """
    Encodes the playlist into the 'training-set basis'. A playlist can only be represented with
    the tracks that are in the training set, as all other tracks are unknown to that playlist.
    """

    def __init__(self, tracks, accumulate):
        """
        Parameters
        ---
            tracks: the list of track ids used to encode a playlist.
        """
        self.mapping = {tid: i for i, (tid, *_) in enumerate(tracks)}
        self.accumulate = accumulate

    # def __call__(self, tids):
    #     track_indices = [self.mapping[tid] for tid in tids if tid in self.mapping]
    #     encoding = np.zeros(len(self))
    #     for index, listens in zip(*np.unique(track_indices, return_counts=True)):
    #         encoding[index] = listens if self.accumulate else 1
    #     return encoding

    def __call__(self, tids):
        return [self.mapping[tid] for tid in tids if tid in self.mapping]

    def __len__(self):
        return len(self.mapping)


class MelEncoder():
    def __init__(self, fp, mean, std, length, min=0, max=1, loader=None):
        self.fp = fp
        self.length = length
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std

    def __call__(self, tid):
        mel = np.load(os.sep.join((self.fp, tid+'.npy')))
        tiled = np.tile(mel, (1, 1 + self.length // mel.shape[1]))[:,:self.length]
        mapped = (tiled - self.min)/(self.max - self.min)
        normalized = (mapped - self.mean)/self.std
        return normalized

    def __len__(self):
        return self.length
