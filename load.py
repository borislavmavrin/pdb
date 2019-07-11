import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from dataloader import *
from utility import *

class Net(nn.Module):
	def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_hidden6, n_hidden7, n_hidden8, n_hidden9, n_output):
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
		self.out = nn.Linear(n_hidden9, n_output)

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
		return self.out(x)

def main():
	model = Net(16, 512, 512, 512, 512, 512, 512, 512, 512, 512, 45)
	# model= nn.DataParallel(model)
	print("loading model ...")
	model.load_state_dict(torch.load("model8-15.txt"))
	print("done")
	# y = readPDB("delta0-4.txt")
	# x = readCoord("coord0-4.txt", 5)
	data = Data("compDelta8-15.txt", [8, 9, 10, 11, 12, 13, 14, 15], 19200)
	print("loading")
	data.load()
	print("done.")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# x = x.to(device)
	# y = y.to(device)
	model.to(device)

	num_wrongPrediction = data.get_num_wrong_prediction(model, device)
	print(1- (num_wrongPrediction/len(data.label)), "  ", num_wrongPrediction, "/", len(data.label))

	# output = model(x)
	# _ , prediction = torch.max(output, 1)
	# print(getAccuracy(prediction, y))
	# print(prediction == y)



	# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	# loss_fn = nn.CrossEntropyLoss()
	# data_size = len(data.label)
	# num_batches = data_size // data.batch_size
	#
	# for epoch in range(100000):
	# 	for i in range(num_batches):
	# 		x, y = data.get_batch()
	# 		x = x.to(device)
	# 		y = y.to(device)
	# 		out = model(x)
	# 		loss = loss_fn(out, y)
	# 		optimizer.zero_grad()
	# 		loss.backward()
	# 		optimizer.step()
	# 		if i % 1000 == 0:
	# 			torch.save(model.state_dict(), "model8-15.txt")
	# 			print(loss, i)
	# 	data.reshuffle()


	# for i in range(20000):
	# 	out = model(x)
	# 	loss = loss_fn(out, y)
	# 	optimizer.zero_grad()
	# 	loss.backward()
	# 	optimizer.step()
	# 	if i%100 == 99:
	# 		print(loss, i+1)
	# 	if i%1500 == 1499:
	# 		output = model(x)
	# 		_, prediction = torch.max(output, 1)
	# 		temp = getAccuracy(prediction, y)
	# 		print(temp)
	# 		if temp == 100:
	# 			break
	#
	# output = model(x)
	# _ , prediction = torch.max(output, 1)
	# print(getAccuracy(prediction, y))
	# print(prediction == y)
	# torch.save(model.state_dict(), "model.txt")

if __name__ == "__main__":
    main()





