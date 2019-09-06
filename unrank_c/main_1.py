import time
import timeit
import pyximport;

pyximport.install()
import unrank_c.unrank


def main():
    start = time.time()
    result = unrank_c.unrank.un(1, list(range(6)))
    duration = time.time() - start
    print(result, duration)


def main1():
    print(timeit.timeit('unrank_c.unrank.getFeature(1, list(range(6)))', setup='import unrank_c.unrank', number=1000000))
    # print(timeit.timeit('unrank.unrank(1, list(range(6)))', setup='imort unrank', number=1000000))


def main2():
    print(timeit.timeit('dataloader.getFeature(1, list(range(6)))', setup='import dataloader', number=1000000))
    # print(timeit.timeit('unrank.unrank(1, list(range(6)))', setup='imort unrank', number=1000000))


if __name__ == '__main__':
    main2()
    main1()