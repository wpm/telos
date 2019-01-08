from random import choices
from typing import Iterable, Tuple

from cytoolz.itertoolz import take
from numpy import array


def sample_markov_model(transition, emission) -> Iterable[Tuple[int, int]]:
    s = 0
    while True:
        s = choices(range(transition.shape[1]), weights=transition[s].flatten())[0]
        o = choices(range(1, emission.shape[1] + 1), weights=emission[s].flatten())[0]
        yield s, o


def main():
    transition = array([
        [0.0, 0.5, 0.5],
        [0.0, 0.75, 0.25],
        [0.0, 0.25, 0.75]
    ])
    emission = array([
        [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.6, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.6],
    ])

    for _ in range(10):
        print([f"{chr(ord('A') + s)}:{o}" for s, o in take(10, sample_markov_model(transition, emission))])


if __name__ == '__main__':
    main()
