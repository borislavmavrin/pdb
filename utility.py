import numpy as np
import torch

def getAccuracy(prediction, y):
	assert len(prediction) == len(y)
	size = len(prediction)
	count = 0
	for i in range(len(prediction)):
		if prediction[i] == y[i]:
			count += 1
	print(size-count, '/', size)
	return count/size*100

def getAccuracy_underestim(prediction, y):
    assert len(prediction) == len(y)
    size = len(prediction)
    count = 0
    for i in range(len(prediction)):
        if prediction[i] <= y[i]:
            count += 1
    print(size - count, '/', size)
    return count / size * 100

# n is the rank,
# size is the size of pattern
def unrank(n, pattern):
	size = len(pattern)
	s = [-1] * 16
	fixed = [-1] * size
	temp = 17 - size
	for i in range(temp, 16):
		fixed[16-i] = n % i
		n = n // i
	fixed[0] = n

	for i in range(size-2, -1, -1):
		for j in range(i+1, size):
			if fixed[i] <= fixed[j]:
				fixed[j] += 1

	for i in range(size):
		s[fixed[i]] = pattern[i]
	# print(s)
	return s

def getFeature(count, pattern):
	feature = []
	s = unrank(count, pattern)
	dual = [-1] * 16
	for i in range(16):
		if s[i] != -1:
			dual[s[i]] = i
	for j in range(len(pattern)):
		feature.append(dual[pattern[j]]//4)
		feature.append(dual[pattern[j]]%4)
	# print(feature)
	return feature

def my_loss(output, target):
	a = 1
	b = 5
	target = target.float()
	err = (output - target) ** 2
	loss = err * (a + 1. / (1. + (torch.exp(-b * err))))
	loss = loss.mean()
	return loss

# unrank(3, [1,2,3,4,5])


# lst = []
# with open("1-7.txt", "rb") as f:
# 	byte = f.read(1)
# 	count = 1
# 	while byte:
# 		if count > 88:
# 			lst.append(byte)
# 		byte = f.read(1)
# 		count += 1
# 	print(count)

# lst = lst[-150:]
# for i in lst:
# 	print(ord(i), end=" ")


# lst = []
# with open("compDelta1-7.txt", "rb") as f:
# 	byte = f.read(1)
# 	count = 1
# 	while byte:
# 		if count <= 57657600:
# 			lst.append(byte)
# 			byte = f.read(1)
# 			count += 1
# 		else:
# 			break
# 	print(count)
#
# with open("compDelta1-7.txt", "wb") as f:
# 	for i in lst:
# 		f.write(i)


# with open ("compDelta1-7.txt", "rb") as f:
# 	byte = f.read(1)
# 	count = 1
# 	while byte:
# 		byte = f.read(1)
# 		count += 1
# print(count)


