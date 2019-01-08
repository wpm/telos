from random import shuffle, sample
from typing import Tuple, Callable

from numpy import arange, zeros, array


def long_distance_dependencies():
    xs = arange(10)
    ys = zeros(10, int)
    shuffle(xs)
    i, j = sorted(sample(range(10), 2))
    xs[j] = xs[i]
    ys[i] = ys[j] = 1
    return xs, ys


def aligned_sequences(n: int, sequence_sampler: Callable[[], Tuple[array, array]]) -> Tuple[array, array]:
    xs, ys = sequence_sampler()
    x = zeros((n, xs.shape[0]), int)
    y = zeros((n, ys.shape[0]), int)
    for i in range(n):
        xs, ys = sequence_sampler()
        x[i] = xs
        y[i] = ys
    return x, y


def main():
    x, y = aligned_sequences(5, long_distance_dependencies)
    print(x)
    print(y)


if __name__ == '__main__':
    main()
