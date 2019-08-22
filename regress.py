import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from dataloader import *
from utility import *


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_hidden6, n_hidden7,
                 n_hidden8, n_hidden9, n_hidden10, n_hidden11, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = nn.Linear(n_hidden3, n_hidden4)
        self.hidden5 = nn.Linear(n_hidden4, n_hidden5)
        self.hidden6 = nn.Linear(n_hidden5, n_hidden6)
        self.hidden7 = nn.Linear(n_hidden6, n_hidden7)
        self.hidden8 = nn.Linear(n_hidden7, n_hidden8)
        self.hidden9 = nn.Linear(n_hidden8, n_hidden9)
        self.hidden10 = nn.Linear(n_hidden9, n_hidden10)
        self.hidden11 = nn.Linear(n_hidden10, n_hidden11)
        # self.hidden12 = nn.Linear(n_hidden11, n_hidden12)
        self.out = nn.Linear(n_hidden11, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = F.relu(self.hidden10(x))
        x = F.relu(self.hidden11(x))
        # x = F.relu(self.hidden12(x))
        return self.out(x)


def main():
    import os
    home = os.path.expanduser("~")
    data_temp = "compDelta1-7.txt"
    data_path = os.path.join(home, data_temp)
    model_temp = "deltaRegresssModel1-7"
    model_name = model_temp + ".txt"

    model = Net(14, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1)
    # torch.save(model.state_dict(), model_name)
    print("done")

    data = Data(data_path, [1, 2, 3, 4, 5, 6, 7], 11520)
    print("loading", data_path)
    data.load()
    print("done.")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = my_loss
    # loss_fn = nn.MSELoss()

    # if torch.cuda.device_count() > 1:
    # 	print(torch.cuda.device_count(), "GPUs")
    # model = nn.DataParallel(model)

    model.to(device)

    # print("loading data ...")
    # features, labels = readSplitPDB(data_name, [1,2,3,4,5,6,7], split, nth)
    # features = features.to(device)
    # labels = labels.to(device)
    # print("done")

    data_size = len(data.label)
    num_batches = data_size // data.batch_size
    num_batches = 100

    for epoch in range(1):
        for i in range(num_batches):
            x, y = data.get_batch()
            x = x.to(device)
            y = y.to(device)
            out = model(x).squeeze()
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                # torch.save(model.state_dict(), model_name)
                print(loss, i)
        data.reshuffle()


# for i in range(1000000000):
# 	out = model(features)
# 	loss = loss_fn(out, labels)
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()
# 	if i % 1000 == 0:
# 		torch.save(model.state_dict(), model_name)
# 		print(loss, i)

# output = model(x)
# _ , prediction = torch.max(output, 1)
# accu = getAccuracy(prediction, y)
# print(accu)
# print(prediction == y)

# torch.save(model.state_dict(), "test.txt")
# model = model.to('cpu')
# torch.save(model.state_dict(), "modelcpu.txt")
# for i in range(len(temp)):
# 	print(temp[i])


if __name__ == "__main__":
    main()
