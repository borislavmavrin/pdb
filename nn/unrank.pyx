# import numpy as np
# cimport numpy as np


def unrank(n, pattern):
    cdef int n_ = n
    cdef list pattern_ = pattern
    cdef int size = len(pattern_)
    cdef int temp = 17 - size
    cdef list fixed = [-1] * size
    cdef list s = [-1] * 16
    # cdef np.ndarray s = -np.zeros((16,), dtype=np.int)
    cdef int i = 0
    cdef int j = 0

    for i in range(temp, 16):
        fixed[16 - i] = n_ % i
        n_ = n_ // i
    fixed[0] = n_

    for i in range(size - 2, -1, -1):
        for j in range(i + 1, size):
            if fixed[i] <= fixed[j]:
                fixed[j] += 1

    for i in range(size):
        s[fixed[i]] = pattern_[i]
    return s


def getFeature(count, pattern):
    cdef int count_ = count
    cdef list pattern_ = pattern
    cdef list feature = []
    cdef list s = unrank(count_, pattern_)
    cdef list dual = [-1] * 16
    cdef int i = 0
    cdef int j = 0
    for i in range(16):
        if s[i] != -1:
            dual[s[i]] = i
    for j in range(len(pattern_)):
        feature.append(dual[pattern_[j]] // 4)
        feature.append(dual[pattern_[j]] % 4)
    # print(feature)
    return feature
