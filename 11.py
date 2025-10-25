import torch
from torch.utils.data import Dataset
import numpy as np

class CoraDataset(Dataset):
    def __init__(self, data_dir='./data/cora'):
        self.data_dir = data_dir
        self.features, self.labels = self.load_data()
    def load_data(self):
        D = np.genfromtxt('cora.content', dtype=str)
        F = D[:, 1:-1].astype(np.float32)
        Labels = D[:, -1]
        Ulabels = np.unique(Labels)
        label_to_idx = {label: i for i, label in enumerate(Ulabels)}
        labels = np.array([label_to_idx[label] for label in Labels])
        Ftensor = torch.FloatTensor(F)
        Ltensor = torch.LongTensor(labels)
        return Ftensor, Ltensor
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]