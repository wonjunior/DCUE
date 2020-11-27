from fetching import fetch_mpd, fetch_mels
from processing import analyse_mels, processing, split

def download(mpd, mels, limit):
    _, tracks = fetch_mpd(mpd, limit)
    filtered_tracks = [t for t in tracks if len(t[-1]) > 1] # remove tracks with a single occurence
    fails = fetch_mels(filtered_tracks, mels)
    print('failed to find and download %i tracks' % len(fails))


def analyse(mpd, mels, limit, mel_min, mel_max):
    _, tracks = fetch_mpd(mpd, limit)
    track_list = processing(tracks, mels)
    train_tracks, *_ = split(track_list)
    x_mean, x_std = analyse_mels(mels, train_tracks, x_min=mel_min, x_max=mel_max)
    print('mean=%.4f; std=%.4f' % (x_mean, x_std))
