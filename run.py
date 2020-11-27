"""CLI endpoints scripts."""
def main(args):
    from scripts import stats, show_perf, train, mels

    if args.mode == 'stats':
        stats(args.mpd, args.mels, args.limit)

    elif args.mode == 'fetch-mels':
        mels.download(args.mpd, args.mels, args.limit)

    elif args.mode == 'analyse-mels':
        mels.analyse(args.mpd, args.mels, args.limit, args.mel_min, args.mel_max)

    elif args.mode == 'train':
        train(args)

    elif args.mode == 'show':
        show_perf(args.load)


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('mode', type=str, default='train',
        choices=['stats', 'analyse-mels', 'fetch-mels', 'train', 'show'])
    parser.add_argument('--epochs', metavar='NB', type=int, default=50)
    parser.add_argument('-bs', '--batch-size', metavar='NB', type=int, default=10)
    parser.add_argument('-lr', '--learning-rate', metavar='NB', type=float, default=0.2)
    parser.add_argument('--criterion', default='multi_margin', choices=['multi_margin', 'cross_entropy'])
    parser.add_argument('--margin', type=float, default=1, help='Margin for multi-margin loss.')

    parser.add_argument('--mpd', type=str, metavar='FILE', default='../datasets/MPD/subset',
        help='Directory where the Million Playlist Dataset csv files are located.')
    parser.add_argument('--mels', type=str, metavar='FILE', default='../datasets/mels',
        help='Path where the mel-spectrograms are or will be extracted to.')
    parser.add_argument('--load', type=str, metavar='FILE', default=None,
        help='File loaded to resume training, finetune or plot results.')
    parser.add_argument('--save', type=str, metavar='FILE', default='out.pth',
        help='Output file to save model\'s state.')
    parser.add_argument('--nb-negatives', type=int, metavar='NB', default=20,
        help='Size of the track negative sampling.')
    parser.add_argument('--mel-length', metavar='NB', type=int, default=130,#default=1323,
        help='Trims or loops audio signals to ensure all have the same length.')
    parser.add_argument('--embedding-size', metavar='NB', type=int, default=256,
        help='Embedding size of the playlist and tracks.')
    parser.add_argument('--mel-bins', metavar='NB', type=int, default=128,
        help='Number of bins in the mel spectrograms, i.e. number of channels in 1D-CNN.')
    parser.add_argument('--mel-mean', metavar='NB', type=float, default=0.4767,
        help='Mean of the mel spectrogtrams in the training set.')
    parser.add_argument('--mel-std', metavar='NB', type=float, default=0.1784,
        help='Standard deviation of the mel spectrogtrams in the training set.')
    parser.add_argument('--limit', metavar='NB', type=int, default=10000,
        help='Limits the number of playlists to load.')

    # should be in the mel preprocessing
    parser.add_argument('--mel-min', metavar='NB', type=float, default=-80,
        help='Minimum value of the mel-spectrogtram.')
    parser.add_argument('--mel-max', metavar='NB', type=float, default=0,
        help='Maximum value of the mel-spectrogtram.')

    args = parser.parse_args()

    if os.path.isfile(args.save):
        print('warning: file "%s" already exists, you risk losing it' % args.save)

    main(args)
