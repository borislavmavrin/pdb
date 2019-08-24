import numpy as np
from dataloader import *
import os
import pickle


DEBUG = True
data_file = "compDelta1-7.txt"
pattern = [1, 2, 3, 4, 5, 6, 7]
home = os.path.expanduser('~')
data_path = os.path.join(home, 'pdb_data', data_file)
pkl_path = os.path.join(home, 'pdb_data', data_file.split('.')[0] + '.pkl.')
label_lst = []
with open(data_path, "rb") as f:
    byte = f.read(1)
    while byte:
        label_lst.append(ord(byte))
        byte = f.read(1)
data_size = len(label_lst)


y_lst = []
multichannel_img_lst = []
batch_size = data_size // 500
batch = 0
for i, y in enumerate(label_lst):
    y_lst.append(y)
    s = getFeature(i, pattern)
    s_np = np.array(s)
    s_np = s_np.reshape((-1, 2)).astype(int)
    z = np.zeros((7, 4, 4))
    z[np.arange(7), s_np[:, 0], s_np[:, 1]] = 1.
    multichannel_img_lst.append(z)
    if (i + 1) % batch_size == 0:
        print('saving batch: ' + str(batch))
        pickle.dump([np.array(y_lst), np.array(multichannel_img_lst)], open(pkl_path + str(batch), "wb"))
        y_lst = []
        multichannel_img_lst = []
        batch += 1
        if DEBUG:
            break

if not DEBUG:
    print('saving batch: ' + str(batch))
    np.save(pkl_path + str(batch), [np.array(y_lst), np.array(multichannel_img_lst)])
