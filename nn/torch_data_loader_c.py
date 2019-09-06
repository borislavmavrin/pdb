from torch.utils.data import Dataset, DataLoader
from dataloader import *
import numpy as np
import torch
import os
import pyximport
pyximport.install()
import unrank_c.unrank


class MultiChannelDataset(Dataset):

    def __init__(self, pdb_file, pattern, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pattern = pattern
        label_lst = []
        with open(pdb_file, "rb") as f:
            byte = f.read(1)
            while byte:
                label_lst.append(ord(byte))
                byte = f.read(1)
        self.labels = np.array(label_lst)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx].astype('long')
        s = unrank_c.unrank.getFeature(idx, self.pattern)
        s_np = np.array(s).astype('float')
        s_np = s_np.reshape((-1, 2)).astype(int)
        state = np.zeros((7, 4, 4))
        state[np.arange(7), s_np[:, 0], s_np[:, 1]] = 1.
        sample = {'state': state, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    data_file = "compDelta1-7.txt"
    pattern = [1, 2, 3, 4, 5, 6, 7]
    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'pdb_data', data_file)
    multi_channel_dataset = MultiChannelDataset(pdb_file=data_path, pattern=pattern)
    dataloader = DataLoader(multi_channel_dataset, batch_size=4,
                            shuffle=True, num_workers=32)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['state'].size(),
              sample_batched['label'].size())
