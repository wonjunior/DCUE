def profile(fn):
    import cProfile, pstats, io
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = fn(*args, **kwargs)
        pr.disable()
        stream = io.StringIO()
        ps = pstats.Stats(pr, stream=stream).sort_stats('time')
        ps.print_stats()
        print(stream.getvalue())
        return result
    return wrapper


def bar(x, title='', xlabel='', ylabel=''):
    from collections import Counter
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tick
    from statistics import mean

    freq = Counter(sorted(x))
    count = sum(freq.values())
    height = [f/count*100 for f in freq.values()]

    ax = plt.subplot()
    ax.bar(freq.keys(), height, align='center', color='#3c3c3c', alpha=0.5)
    ax.yaxis.set_major_formatter(tick.PercentFormatter())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
