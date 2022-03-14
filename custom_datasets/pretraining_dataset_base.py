import torch, logging, random

class PretrainingDatasetBase(torch.utils.data.Dataset):

    def __init__(self, training_data, **config):

        self.data = training_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        raise NotImplementedError