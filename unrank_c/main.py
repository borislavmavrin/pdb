import time
import timeit
# import pyximport;

# pyximport.install()
import unrank.unrank


def unrank_pure(n, pattern):
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


def main():
    start = time.time()
    result = unrank_pure(1, list(range(6)))
    duration = time.time() - start
    print(result, duration)


def main1():
    # print(timeit.timeit('unrank_pure(1, list(range(6)))', setup='from main import unrank_pure', number=1000000))
    print(timeit.timeit('unrank.unrank(1, list(range(6)))', setup='imort unrank', number=1000000))


if __name__ == '__main__':
    main1()
