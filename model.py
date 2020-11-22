import time
from statistics import mean
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class PlaylistMLP(nn.Module):
    def __init__(self, encoding_size, emb_size):
        super(PlaylistMLP, self).__init__()
        self.body = nn.Sequential(
            nn.Embedding(encoding_size, 300),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, emb_size)
        )

    def forward(self, y):
        return self.body(y)


class MelCNN(nn.Module):
    def __init__(self, cn_in, emb_size):
        super(MelCNN, self).__init__()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels=cn_in, out_channels=emb_size, kernel_size=4, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, padding=2, stride=1),

            nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, padding=2, stride=1),

            nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=4, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, padding=2, stride=1),

            nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=2, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=1),

            nn.Conv1d(in_channels=emb_size, out_channels=1, kernel_size=1, padding=1, stride=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(emb_size),
            nn.Flatten(),
            nn.Linear(emb_size, emb_size)
        )

    def forward(self, y):
        return self.body(y)


class DCUE(nn.Module):
    """Siaseme model run with negative sampling."""

    def __init__(self, playlist_enc_size, cnn_channels, emb_size): # add option to change similarity
        super(DCUE, self).__init__()
        self.mlp = PlaylistMLP(playlist_enc_size, emb_size)
        self.cnn = MelCNN(cn_in=cnn_channels, emb_size=emb_size)
        self.similarity = nn.CosineSimilarity() #eps=1e-6

    def forward(self, playlist, positive, negatives):
        U = self.mlp(playlist.long())
        I_pos = self.cnn(positive.float())
        I_neg = [self.cnn(n.float()) for n in negatives]
        I = torch.stack([I_pos] + I_neg, axis=1)
        sim = [self.similarity(U, I[:,i]) for i in range(I.shape[1])]
        return torch.stack(sim, axis=1)


class DCUEWrapper:
    """Runner wrapper for the DCUE model."""

    def __init__(self, model, train_dl, valid_dl, save, criterion, margin):
        self.save = save
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, nesterov=True)

        if criterion == 'cross_entropy':
            self.criterion = nn.MultiMarginLoss(margin=margin)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs, load=None):
        if load:
            start, losses, since = self.load(load)
            print('Resume training at epoch={} (already elapsed: {})' \
                .format(start + 1, timedelta(seconds=int(time.time()-since))))
        else:
            start, losses, since = 0, [], time.time()

        for epoch in range(start, epochs):
            print('Epoch %i/%i' % (epoch + 1, epochs))
            losses += self.train_epoch()
            elapsed = time.time() - since
            self.checkpoint(epoch, losses, elapsed)
        print('Training finished in {}'.format(timedelta(seconds=int(elapsed))))

    def checkpoint(self, epoch, losses, elapsed):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': losses,
            'elapsed': elapsed,
        }, self.save)

    def load(self, fp):
        print('loading state from %s' % fp)
        checkpoint = torch.load(fp)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        start = time.time() - checkpoint['elapsed']
        return checkpoint['epoch'], checkpoint['losses'], start

    def train_epoch(self):
        losses, accuracies = [], []
        t = tqdm(self.train_dl)
        for x in t:
            self.optimizer.zero_grad()

            y = torch.zeros(x[0].shape[0], dtype=torch.long)
            y_hat = self.model(*x)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            accuracy = (torch.argmax(y_hat, dim=1) == 0).float().mean().item()
            losses.append(loss)
            accuracies.append(accuracy)
            t.set_description('loss {:.5f} (mean={:.5f}) - accuracy {:.1f}% (mean={:.1f}%)' \
                .format(loss, mean(losses), accuracy*100, mean(accuracies)*100))
        return losses

    def validate(self):
        losses = []
        with torch.no_grad():
            for x in self.valid_dl:
                y = torch.zeros(x[0].shape[0], dtype=torch.long)
                y_hat = self.model(*x)
                loss = self.criterion(y_hat, y)
                losses.append(loss.item())
        return losses
