import torch
import numpy as np
import math
from utility import *


class Data:

	def __init__(self, filename, pattern, batch_size):
		self.filename = filename
		self.batch_size = batch_size
		self.pattern = pattern
		self.label = []
		self.y = []
		self.x = []
		self.count = 0
		self.lower = -1
		self.upper = -1
		self.data_size = 0
		self.idx = None

	# def reset(self):
	# 	self.label = []
	# 	self.y = []
	# 	self.x = []
	# 	self.count = 0
	# 	self.lower = -1
	# 	self.upper = -1

	def reshuffle(self):
		np.random.shuffle(self.idx)

	def load(self):
		with open(self.filename, "rb") as f:
			byte = f.read(1)
			while byte:
				self.label.append(ord(byte))
				byte = f.read(1)
		self.data_size = len(self.label)
		self.idx = np.arange(self.data_size)
		np.random.shuffle(self.idx)

	def get_batch(self):
		self.x.clear()
		self.y.clear()
		self.lower = self.count * self.batch_size
		self.upper = (self.count + 1) * self.batch_size
		self.count += 1
		if self.upper >= len(self.label):
			self.upper = len(self.label)
			self.count = 0
		# batch_dx = np.random.choice(self.data_size, size=self.batch_size, replace=False)
		for i in self.idx[range(self.lower, self.upper)]:
			self.y.append(self.label[i])
			self.x.append(getFeature(i, self.pattern))
		return torch.FloatTensor(self.x), torch.LongTensor(self.y)

	def get_batch_rank(self):
		self.x.clear()
		self.y.clear()
		self.lower = self.count * self.batch_size
		self.upper = (self.count + 1) * self.batch_size
		self.count += 1
		if self.upper >= len(self.label):
			self.upper = len(self.label)
			self.count = 0
		for i in self.idx[range(self.lower, self.upper)]:
			self.y.append(self.label[i])
			self.x.append([i])
			return torch.FloatTensor(self.x), torch.LongTensor(self.y)

	def get_num_wrong_prediction(self, model, device):
		ct = 0
		num_wrongPrediction = 0
		np.random.shuffle(self.idx)
		while True:
			temp = num_wrongPrediction
			lower = ct * 10000
			upper = (ct + 1) * 10000
			ct += 1
			if upper < len(self.label):
				x = []
				y = []
				for i in self.idx[range(lower, upper)]:

					y.append(self.label[i])
					x.append(getFeature(i, self.pattern))
					temp_features = torch.FloatTensor(x).to(device)
					temp_labels = torch.LongTensor(y).to(device)
				out = model(temp_features)
				_, prediction = torch.max(out, 1)
				# temp = (out == temp_labels)
				for i in range(len(temp_labels)):
					if prediction[i] > temp_labels[i]:
						num_wrongPrediction += 1

			else:
				upper = len(self.label)
				x = []
				y = []
				for i in self.idx[range(lower, upper)]:
					y.append(self.label[i])
					x.append(getFeature(i, self.pattern))
					temp_features = torch.FloatTensor(x).to(device)
					temp_labels = torch.LongTensor(y).to(device)
				out = model(temp_features)
				_, prediction = torch.max(out, 1)
				# temp = (out == temp_labels)
				for i in range(len(temp_labels)):
					if prediction[i] > temp_labels[i]:
						num_wrongPrediction += 1
				break
			print(num_wrongPrediction - temp, num_wrongPrediction)
		return num_wrongPrediction


def readPDB(filename, pattern):
	y = []
	x = []
	with open(filename, "rb") as f:
		byte = f.read(1)
		count = 0
		while byte:
			y.append([ord(byte)])
			x.append(getFeature(count, pattern))
			byte = f.read(1)
			count += 1
	return torch.FloatTensor(x), torch.FloatTensor(y)
	# return torch.LongTensor(y)

def readSplitPDB(filename, pattern, div, nth):
	mx = int(math.factorial(16)/math.factorial(16 - len(pattern)))
	offset = int(mx/div*nth)
	print(mx, offset)
	y = []
	x = []
	with open(filename, "rb") as f:
		byte = f.read(1)
		count = 0
		while byte:
			y.append([ord(byte)])
			x.append(getFeature(count+offset, pattern))
			byte = f.read(1)
			count += 1
	return torch.FloatTensor(x), torch.LongTensor(y)




def readCoord(filename, numTiles):
	res = []
	with open(filename, "rb") as f:
		byte = f.read(1)
		count = 0
		temp = []
		while byte:
			temp.append(ord(byte))
			count += 1
			if count % (2*numTiles) == 0:
				res.append(temp[:])
				temp.clear()
			byte = f.read(1)
	return torch.FloatTensor(res)




