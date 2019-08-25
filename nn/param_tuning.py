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
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval


class Net(nn.Module):

    def __init__(self, num_kernels, num_labels, num_layers):
        super(Net, self).__init__()
        self.num_kernels = num_kernels
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(7, self.num_kernels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(self.num_kernels, self.num_kernels, kernel_size=2, stride=1)
        self.linear_lst = nn.ModuleList()
        for l in range(self.num_layers):
            self.linear_lst.append(nn.Linear(self.num_kernels, self.num_kernels))
        self.head = nn.Linear(self.num_kernels, self.num_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        for linear in self.linear_lst:
            x = F.relu(linear(x))
        return self.head(x)


def objective(args):
    num_epochs = 5
    lr = float(args['lr'])
    num_kernels = int(args['num_kernels'])
    num_layers = int(args['num_layers'])

    num_labels = 15
    device = 'cuda:1'

    net = Net(num_kernels, num_labels, num_layers).to(device).float()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    data_file = "compDelta1-7.txt"
    pattern = [1, 2, 3, 4, 5, 6, 7]
    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'pdb_data', data_file)

    batch_size = 32 * 312
    multi_channel_dataset = MultiChannelDataset(pdb_file=data_path, pattern=pattern)
    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=96)

    num_batches = len(dataloader)
    log_steps = num_batches // 10

    start_time = time.time()
    for epoch in range(num_epochs):
        logging.info('epoch: %s', epoch)
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % log_steps == 0:
                elapsed_time = time.time() - start_time
                examples_per_second = batch_size * log_steps / elapsed_time
                start_time = time.time()
                scores = net(sample_batched['state'].to(device).float())
                labels = sample_batched['label'].to(device).long()
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
            labels = sample_batched['label'].to(device).long()
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
    logging.info('finished training')

    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size * 2,
                            shuffle=True, num_workers=96)
    logging.info('calculating exact accuracy...')
    exact_accuracy_lst = []
    for i_batch, sample_batched in enumerate(dataloader):
        scores = net(sample_batched['state'].to(device).float())
        labels = sample_batched['label'].to(device).long()
        scores = scores.detach()
        predictions = scores.argmax(dim=1)
        accuracy = labels == predictions
        accuracy = accuracy.cpu().numpy()
        accuracy = accuracy.sum() / accuracy.shape[0]
        exact_accuracy_lst.append(accuracy)
    exact_accuracy = np.mean(exact_accuracy_lst)

    logging.info('exact accuracy: %s', exact_accuracy)

    return -exact_accuracy


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

    params = {
        'lr': hp.uniform('lr', 1e-5, 1e-1),
        'num_kernels': hp.quniform('num_kernels', 5, 100, 1),
        'num_layers': hp.quniform('num_layers', 1, 10, 1),
    }
    space = params

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    print(best)
    print(space_eval(space, best))
