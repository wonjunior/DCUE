import os, random, math
from tqdm import tqdm

def processing(tracks, mels_dir):
    mels = set(os.listdir(mels_dir))
    track_list = []
    for tid, artist, track, occurences in tracks:
        if len(occurences) > 1 and tid + '.npy' in mels:
            track_list.append((tid, artist, track, occurences))
    return track_list


def split(t, train_split=0.7, valid_split=0.1):
    sepA, sepB = int(len(t)*train_split), int(len(t)*(train_split + valid_split))
    return t[:sepA], t[sepA:sepB], t[sepB:]


def analyse_mels(fp, tracks, x_min=0, x_max=1):
    import numpy as np
    x_sum, x_sum2, n = 0, 0, 0
    for tid, *_ in tqdm(tracks):
        mel = (np.load(os.sep.join((fp, tid + '.npy'))).flatten() - x_min) / (x_max - x_min)
        x_sum += mel.sum()
        x_sum2 += (mel**2).sum()
        n += mel.size

    x_mean = x_sum/n
    x_std = math.sqrt(x_sum2/n - x_mean**2)
    print('mean=%.4f; std=%.4f' % (x_mean, x_std))
