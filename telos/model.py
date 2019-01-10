from random import shuffle, sample
from typing import List, Tuple, Callable

from numpy import newaxis
from numpy.core.multiarray import array, arange, zeros


def sequence_to_sequence_model(time_steps: int = 10, features: int = 1, labels: int = 2, units: List[int] = (16,)):
    """
    Compile a layered bidirectional sequence-to-sequence LSTM model.

    :param time_steps: number of time steps in the sequences
    :param features: number of features in each time step of the feature sequence
    :param labels: number of labels per time step in the label sequence
    :param units: list of number of hidden units in each LSTM layer
    :return: compiled model
    """
    from keras.layers import Bidirectional
    from keras import Sequential
    from keras.layers import LSTM, TimeDistributed, Dense

    model = Sequential()
    for layer in units:
        model.add(Bidirectional(LSTM(units=layer, input_shape=(time_steps, features), return_sequences=True)))
    model.add(TimeDistributed(Dense(labels, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def digits_with_repetition_labels() -> Tuple[array, array]:
    """
    Return a random list of 10 digits from 0 to 9. Two of the digits will be repeated. The rest will be unique.
    Along with this list, return a list of 10 labels, where the label is 0 if the corresponding digit is unique and 1
    if it is repeated.

    :return: digits and labels
    """
    n = 10
    xs = arange(n)
    ys = zeros(n, int)
    shuffle(xs)
    i, j = sample(range(n), 2)
    xs[i] = xs[j]
    ys[i] = ys[j] = 1
    return xs, ys


def labeled_sequences(n: int, sequence_sampler: Callable[[], Tuple[array, array]]) -> Tuple[array, array]:
    """
    Create training data for a sequence-to-sequence labeling model.

    The features are an array of size samples * time steps * 1.
    The labels are a one-hot encoding of time step labels of size samples * time steps * number of labels.

    :param n: number of sequence pairs to generate
    :param sequence_sampler: a function that returns two numeric sequences of equal length
    :return: feature and label sequences
    """
    from keras.utils import to_categorical

    xs, ys = sequence_sampler()
    assert len(xs) == len(ys)
    x = zeros((n, len(xs)), int)
    y = zeros((n, len(ys)), int)
    for i in range(n):
        xs, ys = sequence_sampler()
        x[i] = xs
        y[i] = ys
    x = x[:, :, newaxis]
    y = to_categorical(y).astype(int)
    return x, y
