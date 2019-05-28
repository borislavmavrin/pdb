import numpy as np


with open("../pdb.txt", "r") as f:
    lines_lst = f.readlines()
print(lines_lst[0])


def state2coord(state):
    coords = []
    for tile in [0, 1, 2, 3, 4]:
        row = np.where(state == tile)[0][0] // 4
        coords.append(row)
        column = np.where(state == tile)[0][0] % 4
        coords.append(column)
    return coords


states_np = np.array(list(map(lambda l: l.replace('(4x4)', '').split('  ')[0].split(' '), lines_lst))).astype(np.int8)
labels_np = np.array(list(map(lambda l: l.split('  ')[1], lines_lst))).reshape(-1, 1).astype(np.int8)
# np.vectorize(state2coord)(states_np[:10])
coords_np = np.array([state2coord(s) for s in states_np])
coords_label_np = np.concatenate([coords_np, labels_np], axis=1).astype(np.int8)
np.savetxt("../coords_labels.txt", coords_label_np, delimiter=' ', fmt='%u')

