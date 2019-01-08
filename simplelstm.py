from random import shuffle, sample
from typing import Tuple, Callable

from keras import Sequential, Model
from keras.layers import LSTM, TimeDistributed, Dense
from numpy import arange, zeros, array


def sequence_to_sequence_model(time_steps: int) -> Model:
    model = Sequential()
    model.add(LSTM(units=10, input_shape=(time_steps, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def long_distance_dependencies():
    xs = arange(10)
    ys = zeros(10, int)
    shuffle(xs)
    i, j = sample(range(10), 2)
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
    t = x.shape[1]
    x = x.reshape(x.shape[0], t, 1)
    y = y.reshape(y.shape[0], t, 1)
    return x, y


def main():
    def reshape(a):
        return a.reshape(a.shape[0], a.shape[1])

    # Train
    x, y = aligned_sequences(1000, long_distance_dependencies)
    model = sequence_to_sequence_model(x.shape[1])
    model.summary()
    model.fit(x, y, epochs=20, verbose=2)
    # Test
    x, y = aligned_sequences(5, long_distance_dependencies)
    y_ = model.predict(x, verbose=0)
    x, y, y_ = reshape(x), reshape(y), reshape(y_)
    for i in range(x.shape[0]):
        print(' '.join(str(n) for n in x[i]))
        print(' '.join([' ', '*'][n] for n in y[i]))
        print(y_[i])


if __name__ == '__main__':
    main()
