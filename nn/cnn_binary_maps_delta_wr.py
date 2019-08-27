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

    def __init__(self, num_kernels, num_layers):
        super(Net, self).__init__()
        self.num_kernels = num_kernels
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        self.conv1 = nn.Conv2d(7, self.num_kernels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(self.num_kernels, self.num_kernels, kernel_size=2, stride=1)
        self.linear_lst = nn.ModuleList()
        for l in range(self.num_layers):
            self.linear_lst.append(nn.Linear(self.num_kernels, self.num_kernels))
        self.head = nn.Linear(self.num_kernels, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        for linear in self.linear_lst:
            x = F.relu(linear(x))
        return self.head(x)


def wr_loss(prediction, target):
    prediction = prediction.squeeze()
    target = target.squeeze()
    a = 1
    b = 5
    err = (prediction - target) ** 2
    l = err * (a + 1. / (1. + (torch.exp(-b * err))))
    l = l.mean()
    return l


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    num_epochs = 1
    num_samples = 10
    num_quantiles = 11
    device = 'cuda:0'
    num_workers = 16

    net = Net(32, 1).to(device).float()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    data_file = "compDelta1-7.txt"
    pattern = [1, 2, 3, 4, 5, 6, 7]
    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'pdb_data', data_file)

    batch_size = 32 * 312
    multi_channel_dataset = MultiChannelDataset(pdb_file=data_path, pattern=pattern)
    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

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
                theta = net(sample_batched['state'].to(device).float())
                labels = sample_batched['label'].to(device).float()
                theta = theta.detach()
                loss = wr_loss(theta, labels).detach().cpu().numpy()
                # predictions = theta.mean(dim=1)
                predictions = theta.squeeze()
                accuracy = predictions <= labels
                accuracy = accuracy.cpu().numpy()
                accuracy = accuracy.sum() / accuracy.shape[0]
                logging.info('states per second: %s', examples_per_second)
                logging.info('accuracy: %s', accuracy)
                logging.info('loss: %s', loss)
                # logging.info('i_batch: %s', i_batch)

            theta = net(sample_batched['state'].to(device).float())
            optimizer.zero_grad()
            labels = sample_batched['label'].to(device).float()
            loss = wr_loss(theta, labels)
            # logging.info('loss: %s', loss)
            loss.backward()
            optimizer.step()
    logging.info('finished training')

    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size * 2,
                            shuffle=True, num_workers=num_workers)
    logging.info('calculating exact accuracy...')
    negative_counts = 0
    exact_accuracy_lst = []
    for i_batch, sample_batched in enumerate(dataloader):
        theta = net(sample_batched['state'].to(device).float())
        theta = theta.detach()
        labels = sample_batched['label'].to(device).float()
        predictions = theta[:, 1]
        accuracy = predictions <= labels
        accuracy = accuracy.cpu().numpy()
        accuracy = accuracy.sum() / accuracy.shape[0]
        exact_accuracy_lst.append(accuracy)
        negative_counts += (predictions < 0.).detach().cpu().numpy().sum()
    exact_accuracy = np.mean(exact_accuracy_lst)

    logging.info('exact accuracy: %s', exact_accuracy)
    logging.info('negative counts: %s', negative_counts)
