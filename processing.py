import os, random

def processing(tracks, mels_dir):
    mels = set(os.listdir(mels_dir))
    track_list = []
    for i, (tid, artist, track, occurences) in enumerate(tracks):
        if len(occurences) > 1 and tid + '.npy' in mels:
            track_list.append((tid, artist, track, occurences))
    return track_list

def split(t, train_split=0.7, valid_split=0.1, shuffle=False):
    if shuffle:
        random.shuffle(t)
    sepA, sepB = int(len(t)*train_split), int(len(t)*(train_split + valid_split))
    return t[:sepA], t[sepA:sepB], t[sepB:]
