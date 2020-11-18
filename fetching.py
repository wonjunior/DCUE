import os, re, json, requests

import numpy as np
from tqdm import tqdm

def fetch_mpd(fp):
    if not os.path.isdir(fp):
        raise FileNotFoundError('directory "%s" not found' % fp)

    playlists, tracks = [], {}

    for chunk in os.listdir(fp):
        f = open(os.sep.join((fp, chunk)))
        js = f.read()
        f.close()
        mpd_slice = json.loads(js)

        for playlist in mpd_slice['playlists']:
            pid = playlist['pid']
            playlists.append([])

            for t in playlist['tracks']:
                tid = t['track_uri'].replace('spotify:track:', '')
                artist = t['artist_name'].strip()
                track = re.split('[(-]', t['track_name'])[0].strip()

                if tid not in tracks:
                    tracks[tid] = tid, artist, track, []

                tracks[tid][3].append(pid) # pid will not map to index if chunks are unordered!!
                playlists[-1].append(tid)
    return playlists, list(tracks.values())


PAGE_ERROR = '404'
NO_MP3 = 'preview_missing'
def fetch_mel(tid, artist, track, fp):
    if os.path.isfile(fp + '.wav'):
        return fp + '.wav'

    endpoint = 'https://api.deezer.com/search?q=artist:"%s" track:"%s"' % (artist, track)
    req = requests.get(endpoint)
    if req.status_code != 200:
        return PAGE_ERROR, tid, endpoint

    json = req.json()
    if 'data' in json and len(json['data']):
        mp3_url = json['data'][0]['preview']
    else:
        mp3_url = ''

    if not mp3_url:
        return NO_MP3, tid, endpoint

    mp3 = requests.get(mp3_url)
    if mp3.status_code != 200:
        return NO_MP3, tid, endpoint

    with open(fp, 'wb') as f:
        f.write(mp3.content)

    ffmpeg.input(fp).output(fp + '.wav', loglevel='quiet').run()
    os.remove(fp)

    return fp + '.wav'


def fetch_mels(tracks, out):
    import ffmpeg
    import librosa
    from mel import MelSpectrogram

    if not os.path.isdir(out):
        raise FileNotFoundError('directory "%s" not found' % out)

    fails = []
    mels = set(os.listdir(out))
    extract_mel = MelSpectrogram(sr=22050,  n_fft=1024, win_length=1024, hop_length=512, n_mels=128)
    for tid, artist, track, _ in tqdm(tracks):
        if tid + '.npy' in mels:
            continue

        wav = fetch_mel(tid, artist, track, os.sep.join((out, tid)))
        if type(wav) is tuple:
            fails.append(wav)
            continue

        np.save(wav.replace('wav', 'npy'), extract_mel(wav))
        os.remove(wav)
    return fails
