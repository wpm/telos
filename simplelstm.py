from random import shuffle, sample
from typing import Tuple, Callable

from numpy import arange, zeros, array, argmax, newaxis


def sequence_to_sequence_model(time_steps: int, labels: int, units: int = 16):
    from keras import Sequential
    from keras.layers import LSTM, TimeDistributed, Dense

    model = Sequential()
    model.add(LSTM(units=units, input_shape=(time_steps, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(labels, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


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
    y = to_categorical(y)
    return x, y


def digits_with_repetition_labels() -> Tuple[array, array]:
    """
    Return a random list of 10 digits from 0 to 9. Two of the digits will be repeated. The rest will be unique.
    Along with this list, return a list of 10 labels, where the label is 0 if the corresponding digits is unique and 1
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


def main():
    # Train
    x, y = labeled_sequences(10000, digits_with_repetition_labels)
    model = sequence_to_sequence_model(x.shape[1], y.shape[2])
    model.summary()
    model.fit(x, y, epochs=500, verbose=2)
    # Test
    x, y = labeled_sequences(5, digits_with_repetition_labels)
    y_ = model.predict(x, verbose=0)
    x = x[:, :, 0]
    for i in range(x.shape[0]):
        print(' '.join(str(n) for n in x[i]))
        print(' '.join([' ', '*'][int(argmax(n))] for n in y[i]))
        print(' '.join([' ', '*'][int(argmax(n))] for n in y_[i]))
        print()


if __name__ == '__main__':
    main()
