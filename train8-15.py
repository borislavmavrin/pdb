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
	torch.save(model.state_dict(), "model8-15.txt")
	print("done")


	data = Data("compDelta8-15.txt", [8,9,10,11,12,13,14,15], 19200)
	print("loading")
	data.load()
	print("done.")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# model = Net(16, 512, 512, 512, 512, 512, 512, 512, 512, 512, 45)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	loss_fn = nn.CrossEntropyLoss()

	# if torch.cuda.device_count() > 1:
	# 	print(torch.cuda.device_count(), "GPUs")
		# model = nn.DataParallel(model)

	model.to(device)
	# print("loading data ...")
	# features, labels = readPDB("compDelta8-15.txt", [8,9,10,11,12,13,14,15])
	# features = features.to(device)
	# labels = labels.to(device)
	# print("done")

	data_size = len(data.label)
	num_batches = data_size // data.batch_size

	for epoch in range(10):
		with open("epoch8-15.txt", "w") as f:
			f.write(str(epoch))
		for i in range(num_batches):
			x, y = data.get_batch()
			x = x.to(device)
			y = y.to(device)
			out = model(x)
			loss = loss_fn(out, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % 1000 == 0:
				torch.save(model.state_dict(), "model8-15.txt")
				print(loss, i)
		# if i % 1000000000000 == 999999:
		# 	torch.save(model.state_dict(), "model1-7.txt")
		# 	num_wrongPrediction = data.get_num_wrong_prediction(model, device)
		# 	print(num_wrongPrediction / len(data.label), ", ", num_wrongPrediction, "/ ", len(data.label))
		# 	if num_wrongPrediction == 0:
		# 		break
		data.reshuffle()
		# if i%1500 == 1499:
		# 	torch.save(model.state_dict(), "model8-15.txt")
		# 	output = model(features)
		# 	_, prediction = torch.max(output, 1)
		# 	temp = getAccuracy(prediction, y)
		# 	print(temp)
		# 	if temp == 100:
		# 		break

		# for i in range(100000000000):
		# 	for j in range(len(data.label)):
		# 		y = torch.LongTensor([data.label[j]]).to(device)
		# 		x = torch.FloatTensor(getFeature(j, [1, 2, 3, 4])).to(device)
		# 		out = model(x)
		# 		loss = loss_fn(out, y)
		# 		optimizer.zero_grad()
		# 		loss.backward()
		# 		optimizer.step()
		# 	if i % 1000 == 999:
		# 		print(loss, i + 1)
		# 	if i % 15000 == 14999:
		# 		output = model(features)
		# 		_, prediction = torch.max(output, 1)
		# 		temp = getAccuracy(prediction, labels)
		# 		print(temp)
		# 		if temp == 100:
		# 			break

	# torch.save(model.state_dict(), "model1-6.txt")

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