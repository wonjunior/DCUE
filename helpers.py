def bar(tracks):
    from collections import Counter
    from statistics import mean
    import matplotlib.pyplot as plt

    freq = Counter(sorted(tracks))
    plt.bar(freq.keys(), freq.values(), align='center', color='#3c3c3c', alpha=0.5)
    plt.title('Track distribution')
    plt.xlabel('Number of occurencies in playlists')
    plt.ylabel('Number of tracks having that number of occurencies')
    plt.show()
