import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from dataloader import *
from utility import *


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output):
        super(Net, self).__init__()

        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = nn.Linear(n_hidden3, n_hidden4)
        self.hidden5 = nn.Linear(n_hidden4, n_hidden5)
        # self.hidden6 = nn.Linear(n_hidden5, n_hidden6)
        # self.hidden7 = nn.Linear(n_hidden6, n_hidden7)
        # self.hidden8 = nn.Linear(n_hidden7, n_hidden8)
        # self.hidden9 = nn.Linear(n_hidden8, n_hidden9)
        # self.hidden10 = nn.Linear(n_hidden9, n_hidden10)
        # self.hidden11 = nn.Linear(n_hidden10, n_hidden11)
        # self.hidden12 = nn.Linear(n_hidden11, n_hidden12)
        self.out = nn.Linear(n_hidden5, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        # x = F.relu(self.hidden6(x))
        # x = F.relu(self.hidden7(x))
        # x = F.relu(self.hidden8(x))
        # x = F.relu(self.hidden9(x))
        # x = F.relu(self.hidden10(x))
        # x = F.relu(self.hidden11(x))
        # x = F.relu(self.hidden12(x))
        return self.out(x)


def get_quantile(probs_, quantile_level_):
    cdf = np.cumsum(probs_)
    idx = np.arange(probs_.shape[0])
    return idx[cdf > quantile_level_].min()


if __name__ == "__main__":
    # main()
    device = 'cpu'
    pdb_net = Net(14, 512, 512, 512, 512, 512, 15)
    pdb_net.load_state_dict(torch.load('model1-7.txt', map_location=lambda storage, loc: storage))
    pdb_net.to(device)
    pdb_net.eval()

    softmax = nn.Softmax(dim=1)

    batch_size = 1000

    data = Data("compDelta1-7.txt", [1, 2, 3, 4, 5, 6, 7], batch_size)
    print("loading")
    data.load()
    print("done.")

    data_size = len(data.label)
    num_batches = 100
    # q_level = 0.0005
    q_level = 0.5
    # num_batches = data_size // data.batch_size

    accuracy_lst = []
    bias_lst = []
    avg_q_heuristic_lst = []
    avg_heuristic_lst = []
    for i in range(num_batches):
        x, y = data.get_batch()
        x = x.to(device)
        y = y.to(device)
        score = pdb_net(x)
        probs = softmax(score)
        _, prediction = torch.max(probs, 1)

        prediction_np = prediction.detach().cpu().numpy()
        probs_np = probs.cpu().detach().numpy()
        y_np = y.detach().cpu().numpy()
        q_predictions = np.array([get_quantile(p, q_level) for p in probs_np])
        acc = getAccuracy_underestim(q_predictions, y_np)
        accuracy_lst.append(acc)
        bias_lst.append((np.abs(y_np - q_predictions)).mean())
        avg_q_heuristic_lst.append(q_predictions.mean())
        avg_heuristic_lst.append(prediction_np.mean())
    print("Accuracy: " + str(np.mean(accuracy_lst)))
    print("std err of accuracy: " + str(np.std(accuracy_lst) / batch_size))
    print("MAD: " + str(np.mean(bias_lst)))
    print("std of prediction: " + str(q_predictions.std()))
    print("Average quantile based hearistic: " + str(np.mean(avg_q_heuristic_lst)))
    print("Average hearistic: " + str(np.mean(avg_heuristic_lst)))

