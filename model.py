from statistics import mean
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class PlaylistMLP(nn.Module):
    def __init__(self, encoding_size, embedding_size):
        super(PlaylistMLP, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )

    def forward(self, y):
        return self.body(y)


class MelCNN(nn.Module):
    def __init__(self, embedding_size):
        super(MelCNN, self).__init__()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, padding=2, stride=1),

            nn.Conv1d(in_channels=128, out_channels=embedding_size, kernel_size=4, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, padding=2, stride=1),

            nn.Conv1d(in_channels=128, out_channels=embedding_size, kernel_size=4, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, padding=2, stride=1),

            nn.Conv1d(in_channels=128, out_channels=embedding_size, kernel_size=2, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1, stride=1),

            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, padding=1, stride=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(embedding_size),
            nn.Flatten(),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, y):
        return self.body(y)


class DCUE(nn.Module):
    def __init__(self, playlist_enc_size, embedding_size): # add option to change similarity
        super(DCUE, self).__init__()
        self.mlp = PlaylistMLP(playlist_enc_size, embedding_size)
        self.cnn = MelCNN(embedding_size)
        self.similarity = nn.CosineSimilarity()

    def forward(self, playlist, positive, negatives):
        U = self.mlp(playlist.float())
        I_pos = self.cnn(positive.float())
        I_neg = [self.cnn(n.float()) for n in negatives]
        I = torch.stack([I_pos] + I_neg, axis=1)
        return torch.stack([self.similarity(U, I[:,i]) for i in range(I.shape[1])], axis=1)


class DCUEWrapper:
    """Runner wrapper for the DCUE model."""

    def __init__(self, model, train_dl, valid_dl):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
        self.criterion = nn.MultiMarginLoss(margin=0.2)
        self.fp = 'run.pth'

    def train(self, epochs, load=None):
        if load:
            losses, since = self.load()
        else:
            losses = []
            since = time.time()

        for epoch in range(epochs):
            print('Epoch %i/%i' % (epoch+1, epochs))
            losses += self.train_epoch()
            elapsed = time.time() - since
            self.checkpoint(epoch, losses, elapsed)

            validation_loss = self.validate()
            print('summary of epoch: training loss={:.4f} - validation loss={:.4f}' \
                .format(mean(losses), mean(validation_loss)))
        print('Training finished in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))

    def checkpoint(self, epoch, losses, elapsed):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': losses,
            'elapsed': elapsed
        }, self.fp)

    def load(self):
        checkpoint = torch.load(self.fp)
        start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['losses'], checkpoint['elapsed']

    def train_epoch(self):
        losses = []
        for x in (t := tqdm(self.train_dl)):
            self.optimizer.zero_grad()

            y = torch.zeros(x[0].shape[0], dtype=torch.long)
            y_hat = self.model(*x)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            accuracy = (torch.argmax(y_hat, dim=1) == 0).float().mean().item()
            losses.append(loss)
            t.set_description('loss {:.4f} - accuracy {:.0f}%'.format(loss, accuracy*100))
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
