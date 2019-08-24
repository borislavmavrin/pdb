import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import *
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from nn.torch_data_loader import MultiChannelDataset
import time
import logging


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        num_kernels = 32
        self.conv1 = nn.Conv2d(7, num_kernels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(num_kernels, num_kernels, kernel_size=2, stride=1)
        self.linear1 = nn.Linear(num_kernels * 1 * 1, num_kernels)
        self.head = nn.Linear(num_kernels, 23)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.linear1(x.view(x.shape[0], -1)))
        return self.head(x)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    num_epochs = 10
    num_samples = 10
    device = 'cuda:1'

    net = Net().to(device).float()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    data_file = "compDelta1-7.txt"
    pattern = [1, 2, 3, 4, 5, 6, 7]
    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'pdb_data', data_file)

    batch_size = 32 * 312
    multi_channel_dataset = MultiChannelDataset(pdb_file=data_path, pattern=pattern)
    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=64)

    num_batches = len(dataloader)
    log_steps = num_batches // 10

    epoch = 1
    start_time = time.time()
    for epoch in range(num_epochs):
        logging.info('epoch: %s', epoch)
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % log_steps == 0:
                elapsed_time = time.time() - start_time
                examples_per_second = batch_size * log_steps / elapsed_time
                start_time = time.time()
                scores = net(sample_batched['state'].to(device).float())
                labels = sample_batched['label'].to(device)
                scores = scores.detach()
                predictions = scores.argmax(dim=1)
                accuracy = labels == predictions
                accuracy = accuracy.cpu().numpy()
                accuracy = accuracy.sum() / accuracy.shape[0]
                logging.info('states per second: %s', examples_per_second)
                logging.info('accuracy: %s', accuracy)
                # logging.info('i_batch: %s', i_batch)

            scores = net(sample_batched['state'].to(device).float())
            optimizer.zero_grad()
            loss = criterion(scores, sample_batched['label'].to(device))
            loss.backward()
            optimizer.step()
    logging.info('finished training')

