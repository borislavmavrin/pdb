import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from dataloader import *
from utility import *

def main():
    import os
    home = os.path.expanduser("~")
    data_temp = "compDelta1-7.txt"
    data_path = os.path.join(home, data_temp)

    data = Data(data_path, [1, 2, 3, 4, 5, 6, 7], 11520)
    print("loading", data_path)
    data.load()
    print("done.")

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
