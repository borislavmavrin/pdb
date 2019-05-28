import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 4)
        self.conv2 = nn.Conv2d(4, 4, 3)
        self.fc9 = nn.Linear(4 * 4 * 3, 120)
        self.fc10 = nn.Linear(120, 23)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
        return x


images_np = np.load("../images.npy").astype(np.float32)
labels_np = np.load("../labels.npy").astype(np.float32)
# normalization
# std_np = coords_labels_np[:, :10].std(axis=0)
# means_np = coords_labels_np[:, :10].mean(axis=0)
# coords_labels_np[:, :10] = (coords_labels_np[:, :10] - means_np) / std_np

data_size = images_np.shape[0]
batch_size = 128
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
        coords_tr = torch.from_numpy(images_np[batch_idx]).float().to(device)
        labels_tr = torch.from_numpy(labels_np[batch_idx]).long().to(device)

        optimizer.zero_grad()
        scores = net(coords_tr)
        loss = criterion(scores, labels_tr)
        loss.backward()
        optimizer.step()

    batch_idx = np.random.randint(0, data_size, batch_size)
    coords_tr = torch.from_numpy(images_np[batch_idx]).float().to(device)
    labels_tr = torch.from_numpy(labels_np[batch_idx]).long().to(device)

    scores = net(coords_tr).detach()
    # loss = criterion(scores, labels_tr).detach().cpu()
    _, predicted = torch.max(scores.data, 1)
    print((predicted == labels_tr).sum().item() / data_size)
