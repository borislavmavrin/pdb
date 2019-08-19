import numpy as np


def state2multiChImg(state):
    rows = []
    cols = []
    for tile in [0, 1, 2, 3]:
        row = np.where(state == tile)[0][0] // 4
        rows.append(row)
        column = np.where(state == tile)[0][0] % 4
        cols.append(column)

    rows = np.array(rows)
    cols = np.array(cols)
    level = np.arange(5)
    z = np.zeros((5, 4, 4))
    z[level, rows, cols] = 1
    return z


if __name__ == '__main__':
    s = [3., 0., 2., 3., 3., 3., 3., 1., 2., 1., 0., 1., 1., 2.]
    s_np = np.array(s)
    s_np = s_np.reshape((-1, 2)).astype(int)
    z = np.zeros((2, 7, 4, 4))
    z[np.arange(7), s_np[:, 0], s_np[:, 1]] = 1
    print(z)



    # with open("../data/pdb.txt", "r") as f:
    #     lines_lst = f.readlines()
    # print(lines_lst[0])
    #
    # states_np = np.array(list(map(lambda l: l.replace('(4x4)', '').split('  ')[0].split(' '), lines_lst))).astype(np.int8)
    # labels_np = np.array(list(map(lambda l: l.split('  ')[1], lines_lst))).reshape(-1, 1).astype(np.int8)
    #
    # multi_ch_images_np = np.array([state2multiChImg(s) for s in states_np])
    # np.save("../data/multi_ch_images.npy", multi_ch_images_np)
    # np.save("../data/labels.npy", labels_np)


