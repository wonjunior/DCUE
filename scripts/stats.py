from statistics import mean
from fetching import fetch_mpd
from processing import processing
from utils import bar

def stats(mpd, mels, limit):
    playlists, tracks = fetch_mpd(mpd, limit)
    size = [len(p) for p in playlists]
    tracks = processing(tracks, mels)
    occurencies = [len(t[-1]) for t in tracks]

    print('%i playlists and %i unique tracks' % (len(playlists), len(tracks)))
    print('Playlist sizes: min=%i ; avg=%.2f ; max=%i' % (min(size), mean(size), max(size)))
    print('Track occurences: min=%i ; avg=%.2f ; max=%i' % \
        (min(occurencies), mean(occurencies), max(occurencies)))
    print('Number of interaction (playlist-track pairs): %i' % sum(occurencies))

    # bar(size, title='Playlist sizes', xlabel='Number of tracks', ylabel='freq.')
    # bar(occurencies, title='Track occurencies', xlabel='Number of occurencies', ylabel='freq.')

