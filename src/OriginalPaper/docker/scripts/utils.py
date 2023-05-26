import matplotlib.pyplot as plt

from collections import namedtuple


mri_sample = namedtuple("mri_sample", ("img", "coord", "valid_slices"))

class train_tracker:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.lr = []

    def __len__(self):
        return len(self.train_losses)

    def append(self, train_loss, test_loss, lr):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.lr.append(lr)

    def plot(self, N=None):
        N = N if N is not None else self.__len__()
        plt.plot(self.train_losses[-N:], label='Train')
        plt.plot(self.test_losses[-N:], label='Eval')
        plt.legend()
        plt.show()
