import torch.nn as nn
from dataloader import *
import os
from torch.utils.data import Dataset, DataLoader
from nn.torch_data_loader import MultiChannelDataset
import time
import logging
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from nn.cnn_binary_maps_delta_qr import Net, qr_loss


def objective(args):
    lr = float(args['lr'])
    num_kernels = int(args['num_kernels'])
    num_layers = int(args['num_layers'])
    q_level_idx = int(args['q_level_idx'])

    num_epochs = 5
    num_quantiles = 11
    device = 'cuda:1'
    num_workers = 64

    net = Net(num_kernels, num_layers, num_quantiles).to(device).float()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    data_file = "compDelta1-7.txt"
    pattern = [1, 2, 3, 4, 5, 6, 7]
    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'pdb_data', data_file)

    batch_size = 32 * 312
    multi_channel_dataset = MultiChannelDataset(pdb_file=data_path, pattern=pattern)
    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)

    for epoch in range(num_epochs):
        logging.info('epoch: %s', epoch)
        for i_batch, sample_batched in enumerate(dataloader):
            theta = net(sample_batched['state'].to(device).float())
            optimizer.zero_grad()
            labels = sample_batched['label'].to(device).float()
            loss = qr_loss(theta, labels, device)
            # logging.info('loss: %s', loss)
            loss.backward()
            optimizer.step()
    logging.info('finished training')

    dataloader = DataLoader(multi_channel_dataset, batch_size=batch_size * 2,
                            shuffle=True, num_workers=num_workers)
    logging.info('calculating exact accuracy...')
    exact_accuracy_lst = []
    avg_heuristic_lst = []
    for i_batch, sample_batched in enumerate(dataloader):
        labels = sample_batched['label'].to(device).float()
        theta = net(sample_batched['state'].to(device).float())
        theta = theta.detach()
        # predictions = theta.mean(dim=1)
        predictions = theta[:, q_level_idx]
        accuracy = predictions <= labels
        accuracy = accuracy.cpu().numpy()
        accuracy = accuracy.sum() / accuracy.shape[0]
        exact_accuracy_lst.append(accuracy)

        predictions = predictions.cpu().numpy()
        predictions[predictions < 0.] = 0.
        avg_heuristic_lst.append(predictions.mean())
    exact_accuracy = np.mean(exact_accuracy_lst)
    avg_heuristic = np.mean(avg_heuristic_lst)
    if avg_heuristic <= 2.:
        exact_accuracy = 0.
    logging.info('exact accuracy: %s', exact_accuracy)

    return -exact_accuracy


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

    params = {
        'lr': hp.uniform('lr', 1e-5, 1e-1),
        'num_kernels': hp.quniform('num_kernels', 5, 100, 1),
        'num_layers': hp.quniform('num_layers', 1, 10, 1),
        'q_level_idx': hp.quniform('q_level_idx', 0, 5, 1)
    }
    space = params

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50)

    print(best)
    print(space_eval(space, best))
