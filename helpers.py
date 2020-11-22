from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from statistics import mean

def bar(x, title='', xlabel='', ylabel=''):
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


def show_perf(state, source):
    nb_updates = len(state['losses'])
    nb_batches = nb_updates // state['epoch']
    losses = state['losses']
    avg_losses = [mean(losses[i:i+nb_batches]) for i in range(0, nb_updates, nb_batches)]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 7))
    fig.suptitle('Plotting from %s' % source, fontsize=11)
    ax1.plot(losses, linewidth=0.2, color='black')
    ax1.set_title('Training loss', fontsize=10)
    ax2.plot(avg_losses, linewidth=1, color='black')
    ax2.set_title('Training loss averaged over epoch', fontsize=10)
    plt.show()
