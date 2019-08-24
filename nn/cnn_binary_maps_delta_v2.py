import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import *
import pickle
import os

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


# batch_size = 5000
# data = Data("compDelta1-7.txt", [1, 2, 3, 4, 5, 6, 7], batch_size)
# data.load()

# data_size = len(data.label)
# num_batches = data_size // data.batch_size
# data.reshuffle()

# data_size = images_np.shape[0]
num_epochs = 1
num_samples = 10
device = 'cuda:1'

data_file = "compDelta1-7.txt"
pattern = [1, 2, 3, 4, 5, 6, 7]
home = os.path.expanduser('~')
data_path = os.path.join(home, 'pdb_data', data_file)
pkl_path = os.path.join(home, 'pdb_data', data_file.split('.')[0] + '.pkl.')

net = Net().to(device).float()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
# num_batches = data_size // batch_size
batch_size = 1024 * 10
for epoch in range(num_epochs):
    print(epoch)
    for p in range(5):
        print(p)
        y, x = pickle.load(open(pkl_path + str(0), "rb"))
        p_size = y.shape[0]
        num_batches = p_size // batch_size
        for batch in range(num_batches):
            y_np = y[:batch_size]
            s_np = x[:batch_size]
            # s_np = s_np.reshape((-1, 2)).astype(int)
            # z = np.zeros((batch_size, 7, 4, 4))
            # z[np.repeat(np.arange(batch_size), 7), np.tile(np.arange(7), batch_size), s_np[:, 0], s_np[:, 1]] = 1.
            images_batch_tr = torch.from_numpy(s_np).to(device).float()
            labels_tr = torch.from_numpy(y_np).to(device).long()
            optimizer.zero_grad()
            scores = net(images_batch_tr)
            loss = criterion(scores, labels_tr)
            loss.backward()
            optimizer.step()
    # accuracy_lst = []
    # for s in range(num_samples):
    #     x, y = data.get_batch()
    #     y_np = np.array(y)
    #     s_np = np.array(x)
    #     s_np = s_np.reshape((-1, 2)).astype(int)
    #     z = np.zeros((batch_size, 7, 4, 4))
    #     z[np.repeat(np.arange(batch_size), 7), np.tile(np.arange(7), batch_size), s_np[:, 0], s_np[:, 1]] = 1.
    #     images_batch_tr = torch.from_numpy(z).float()
    #     scores = net(images_batch_tr)
    #     predictions = scores.argmax(dim=1)
    #     labels_tr = torch.from_numpy(y_np).long()
    #
    #     accuracy = np.sum(labels_tr.detach().cpu().numpy() == predictions.detach().cpu().numpy())
    #     accuracy /= y_np.shape[0]
    #     accuracy_lst.append(accuracy)
    # print(np.mean(accuracy_lst))
    # data.reshuffle()
# print(np.max(lables_lst))


