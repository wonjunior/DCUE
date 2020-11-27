from time import time
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

    def __init__(self, playlist_count, cnn_channels, emb_size): # add option to change similarity
        super(DCUE, self).__init__()
        self.mlp = PlaylistMLP(playlist_count, emb_size)
        self.cnn = MelCNN(cn_in=cnn_channels, emb_size=emb_size)
        self.similarity = nn.CosineSimilarity(eps=1e-6)

    def forward(self, playlist, positive, negatives):
        U = self.mlp(playlist.long())
        I_pos = self.cnn(positive.float())
        I_neg = [self.cnn(n.float()) for n in negatives]
        I = torch.stack([I_pos] + I_neg, axis=1)
        sim = [self.similarity(U, I[:,i]) for i in range(I.shape[1])]
        return torch.stack(sim, axis=1)

# class Training: SOON

class DCUEWrapper:
    """Runner wrapper for the DCUE model."""

    def __init__(self, model, train_dl, valid_dl, save, criterion, margin, lr):
        self.fp_out = save
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

        if criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MultiMarginLoss(margin=margin)

    def fit(self, epochs, load=None):
        if load:
            since, start, train_loss, train_acc, valid_loss, valid_acc = self.load(load)
            print('Resume training at epoch={} (already elapsed: {})' \
                .format(start + 1, timedelta(seconds=int(time()-since))))
        else:
            since, start, train_loss, train_acc, valid_loss, valid_acc = time(), 0, [], [], [], []

        for epoch in range(start, epochs):
            print('Epoch %i/%i' % (epoch + 1, epochs))
            elapsed = time() - since
            train = self.train()
            train_loss += train[0]
            train_acc += train[1]
            valid = self.validate()
            valid_loss += valid[0]
            valid_acc += valid[1]
            self.save(elapsed, epoch, train_loss, train_acc, valid_loss, valid_acc)
        print('Training finished in {}'.format(timedelta(seconds=int(elapsed))))

    def save(self, elapsed, epoch, train_loss, train_acc, valid_loss, valid_acc):
        torch.save({
            'elapsed': elapsed,
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
        }, self.fp_out)

    def load(self, fp):
        state = torch.load(fp)
        print('loaded state from %s' % fp)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        start = time() - state['elapsed']
        return start, state['epoch'], state['train_loss'], state['train_acc'], \
               state['valid_loss'], state['valid_acc']

    def train(self):
        losses, accuracies = [], []
        for x in (t := tqdm(self.train_dl)):
            self.optimizer.zero_grad()

            y = torch.zeros(x[0].shape[0], dtype=torch.long)
            y_hat = self.model(*x)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            accuracy = (torch.argmax(y_hat, axis=1) == 0).float().mean()
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            t.set_description('loss {:.5f} (mean={:.5f}) - accuracy {:.1f}% (mean={:.1f}%)' \
                .format(loss, mean(losses), accuracy*100, mean(accuracies)*100))
        return losses, accuracies

    @torch.no_grad()
    def validate(self):
        losses, accuracies = [], []
        for x in self.valid_dl:
            y = torch.zeros(x[0].shape[0], dtype=torch.long)
            y_hat = self.model(*x)

            loss = self.criterion(y_hat, y)
            accuracy = (torch.argmax(y_hat, axis=1) == 0).float().mean()
            losses.append(loss.item())
            accuracies.append(accuracy.item())
