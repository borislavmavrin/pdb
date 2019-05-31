import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        num_kernels = 32
        self.conv1 = nn.Conv2d(5, num_kernels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(num_kernels, num_kernels, kernel_size=2, stride=1)
        self.linear1 = nn.Linear(num_kernels * 1 * 1, num_kernels)
        self.head = nn.Linear(num_kernels, 23)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.linear1(x.view(x.shape[0], -1)))
        return self.head(x)


images_np = np.load("../data/multi_ch_images.npy").astype(np.float32)
labels_np = np.load("../data/labels.npy").astype(np.float32)

data_size = images_np.shape[0]
batch_size = 32
num_epochs = 10000
device = 'cpu'

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
num_batches = data_size // batch_size
for epoch in range(num_epochs):
    print(epoch)
    for batch in range(num_batches):
        batch_idx = np.random.randint(0, data_size, batch_size)
        images_batch_tr = torch.from_numpy(images_np[batch_idx]).float().to(device)
        labels_tr = torch.from_numpy(labels_np[batch_idx]).long().to(device).squeeze()

        optimizer.zero_grad()
        scores = net(images_batch_tr)
        loss = criterion(scores, labels_tr)
        loss.backward()
        optimizer.step()

    batch_idx = np.random.randint(0, data_size, 10000)
    images_batch_tr = torch.from_numpy(images_np[batch_idx]).float().to(device)
    labels_tr = torch.from_numpy(labels_np[batch_idx]).long().to(device).squeeze()

    scores = net(images_batch_tr).detach()
    # loss = criterion(scores, labels_tr).detach().cpu()
    _, predicted = torch.max(scores.data, 1)
    print((predicted == labels_tr).sum().item() / 10000)
