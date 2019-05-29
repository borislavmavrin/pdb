import numpy as np


with open("../data/pdb.txt", "r") as f:
    lines_lst = f.readlines()
print(lines_lst[0])


states_np = np.array(list(map(lambda l: l.replace('(4x4)', '').split('  ')[0].split(' '), lines_lst))).astype(np.int8)
labels_np = np.array(list(map(lambda l: l.split('  ')[1], lines_lst))).reshape(-1, 1).astype(np.int8)
# np.vectorize(state2coord)(states_np[:10])
images_np = states_np.reshape((-1, 4, 4))
np.save("../data/images.npy", images_np)
np.save("../data/labels.npy", labels_np)


