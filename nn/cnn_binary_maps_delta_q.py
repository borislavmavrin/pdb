import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import *
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from nn.torch_data_loader_c import MultiChannelDataset
import time
import logging


class Net(nn.Module):

    def __init__(self, num_kernels, num_layers, num_labels):
        super(Net, self).__init__()
        self.num_kernels = num_kernels
        self.num_layers = num_layers
        self.num_labels = num_labels
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


def get_num_weights(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_weights_ = np.sum([np.prod(p.size()) for p in model_parameters])
    return int(num_weights_)


def get_quantile(probs_, quantile_level_):
    cdf = np.cumsum(probs_)
    idx = np.arange(probs_.shape[0])
    return idx[cdf >= quantile_level_].min()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    lr = 0.01
    num_kernels = 32
    num_layers = 1
    q_level = 0.45
    num_epochs = 1

    device = 'cuda:1'
    num_workers = 64

    net = Net(num_kernels, num_layers, 15).to(device).float()
    softmax = nn.Softmax(dim=1)

    num_weights = get_num_weights(net)
    logging.info('number of weights: %s', num_weights)

    criterion = nn.CrossEntropyLoss()
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
                loss = criterion(scores, labels)
                loss = loss.detach().cpu().numpy()

                scores = scores.detach()
                # predictions = scores.argmax(dim=1)
                # accuracy = labels == predictions
                probs = softmax(scores)
                probs = probs.cpu().numpy()
                q_predictions = np.array([get_quantile(p, q_level) for p in probs])

                accuracy = q_predictions <= labels.detach().cpu().numpy()
                accuracy = accuracy.sum() / accuracy.shape[0]

                avg_heuristic_hat = q_predictions.mean()
                avg_heuristic = labels.cpu().numpy().mean()

                logging.info('states per second: %s', examples_per_second)
                logging.info('accuracy: %s', accuracy)
                logging.info('average predicted heuristic: %s', avg_heuristic_hat)
                logging.info('average heuristic: %s', avg_heuristic)
                logging.info('loss: %s', loss)
                # logging.info('i_batch: %s', i_batch)

            scores = net(sample_batched['state'].to(device).float())
            optimizer.zero_grad()
            labels = sample_batched['label'].to(device).long()
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
    logging.info('finished training')

    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size * 10,
                            shuffle=True, num_workers=num_workers)
    logging.info('calculating exact accuracy...')
    exact_accuracy_lst = []
    avg_heuristic_hat_lst = []
    for i_batch, sample_batched in enumerate(dataloader):
        scores = net(sample_batched['state'].to(device).float())
        labels = sample_batched['label'].to(device).long()
        scores = scores.detach()
        # predictions = scores.argmax(dim=1)
        # accuracy = labels == predictions
        probs = softmax(scores)
        probs = probs.cpu().numpy()
        q_predictions = np.array([get_quantile(p, q_level) for p in probs])

        accuracy = q_predictions <= labels.detach().cpu().numpy()
        accuracy = accuracy.sum() / accuracy.shape[0]
        exact_accuracy_lst.append(accuracy)

        avg_heuristic_hat = q_predictions.mean()
        avg_heuristic_hat_lst.append(avg_heuristic_hat)
        avg_heuristic = labels.cpu().numpy().mean()

        # logging.info('accuracy: %s', accuracy)
        # logging.info('average predicted heuristic: %s', avg_heuristic_hat)
        # logging.info('average heuristic: %s', avg_heuristic)

    exact_accuracy = np.mean(exact_accuracy_lst)
    exact_avg_heuristic = np.mean(avg_heuristic_hat_lst)

    logging.info('exact accuracy: %s', exact_accuracy)
    logging.info('avg predicted heuristic: %s', exact_avg_heuristic)

    # {'num_layers': 2.0, 'num_kernels': 46.0, 'lr': 0.010392954780225002, 'q_level': 0.025473899900400707}
