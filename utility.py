import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import torch



class Mydataset(Dataset):
    def __init__(self, annotations_file, spec_dir, transform=None, target_transform=None):
        self.audio_labels = pd.read_csv(annotations_file).astype('int')
        self.spec_dir = spec_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)
    
    def get_len(self):
        return self.__len__()

    def __getitem__(self, idx):
        spec_path = os.path.join(self.spec_dir, str(self.audio_labels.iloc[idx, 0]) + '.wav.npy',)
        spec = np.load(spec_path)
        label = self.audio_labels.iloc[idx, 1]
        if self.transform:
            spec = self.transform(spec)
        if self.target_transform:
            label = self.target_transform(label)
        return spec, label

def train_test_split(data, proportion = 0.75):
    l = data.get_len()
    return random_split(data, [int(l*proportion), l - int(l*proportion) ])

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X).view(-1)
        y = y.float()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).view(-1)
            y = y.float()
            test_loss += loss_fn(pred, y).item()
            correct += pred.round().sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
