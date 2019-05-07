import numpy as np
import copy


class SlidingTilePuzzle(object):
    def __init__(self, puzzle_size=9):
        self.puzzle_size = puzzle_size
        self.goal_state = np.arange(self.puzzle_size, dtype=np.int8)
        # self.start_state = start_state
        # self.current_state = self.start_state
        self.swapable_idx_lst = [
            [1, 3],
            [0, 2, 4],
            [1, 5],
            [0, 4, 6],
            [1, 3, 5, 7],
            [2, 4, 8],
            [3, 7],
            [4, 6, 8],
            [7, 5]
        ]

    def next_states(self, current_state):
        # identify the position of 0
        zero_idx = np.argwhere(current_state == 0).squeeze()
        next_s_lst_ = []
        for idx in self.swapable_idx_lst[zero_idx]:
            next_s_ = np.copy(current_state)
            next_s_[zero_idx] = current_state[idx]
            next_s_[idx] = current_state[zero_idx]
            next_s_lst_.append(next_s_)
        return next_s_lst_

    def _swap(self, s_, i, j):
        s_copy_ = copy.copy(s_)
        tmp = s_copy_[j]
        s_copy_[j] = s_[i]
        s_copy_[i] = tmp
        return s_copy_

    def _rank(self, n, s_, s_inv):
        if n == 1:
            return 0
        tmp = s_[n - 1]
        s_ = self._swap(s_, n - 1, s_inv[n - 1])
        s_inv = self._swap(s_inv, tmp, n - 1)
        print(tmp)
        return tmp + n * self._rank(n - 1, s_, s_inv)

    def rank(self, s_):
        s_inv = [i for i in s_]
        rank_ = self._rank(self.puzzle_size, s_, s_inv)
        return rank_


if __name__ == '__main__':
    stp = SlidingTilePuzzle()
    s = np.random.choice(9, 9, replace=False)
    # print(s.reshape((3, 3)))
    # print('-' * 8)
    next_s_lst = stp.next_states(s)
    for next_s in next_s_lst:
        pass
        # print(next_s.reshape(3, 3))
    print(stp.rank(list(range(stp.puzzle_size))))
