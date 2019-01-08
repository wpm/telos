from functools import partial
from random import choices
from typing import Iterable, Tuple, Callable

from numpy.core.multiarray import array, zeros


def sample_markov_model(n: int, transition: array, emission: array) -> Iterable[Tuple[int, int]]:
    s = 0
    for _ in range(n):
        s = choices(range(transition.shape[1]), weights=transition[s].flatten())[0]
        o = choices(range(1, emission.shape[1] + 1), weights=emission[s].flatten())[0]
        yield s, o


def markov_samples():
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

    return partial(sample_markov_model, transition=transition, emission=emission)


def aligned_sequences(n: int, t: int, samples: Callable[[int], Iterable[Tuple[int, int]]]) -> Tuple[array, array]:
    states = zeros((n, t), int)
    observations = zeros((n, t), int)
    for i in range(n):
        xs = list(samples(t))
        states[i] = [x[0] for x in xs]
        observations[i] = [x[1] for x in xs]
    return states, observations
