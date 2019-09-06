def unrank(n, pattern):
    size = len(pattern)
    s = [-1] * 16
    fixed = [-1] * size
    temp = 17 - size
    for i in range(temp, 16):
        fixed[16 - i] = n % i
        n = n // i
    fixed[0] = n

    for i in range(size - 2, -1, -1):
        for j in range(i + 1, size):
            if fixed[i] <= fixed[j]:
                fixed[j] += 1

    for i in range(size):
        s[fixed[i]] = pattern[i]
    # print(s)
    return s
