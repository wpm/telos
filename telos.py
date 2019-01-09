from pathlib import Path
from random import shuffle, sample
from typing import Tuple, Callable, Optional

import click
from keras.callbacks import ModelCheckpoint
from numpy import arange, zeros, array, newaxis, sum, size


def sequence_to_sequence_model(time_steps: int = 10, labels: int = 2, units: int = 16):
    from keras.layers import Bidirectional
    from keras import Sequential
    from keras.layers import LSTM, TimeDistributed, Dense

    model = Sequential()
    model.add(Bidirectional(LSTM(units=units, input_shape=(time_steps, 1), return_sequences=True)))
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


@click.group('telos')
def telos_command():
    """
    Deep Learning for Long Distance Dependencies
    """
    pass


@telos_command.command('train')
@click.argument('model_path', metavar='model', type=click.Path(dir_okay=False, readable=True))
@click.option('--units', default=16, show_default=True, help='hidden units in the model')
@click.option('--samples', default=10000, show_default=True, help='number of training samples')
@click.option('--epochs', default=200, show_default=True, help='number of training epochs')
@click.option('--validation-split', type=float, help='portion of data to use for validation')
@click.option('--checkpoint', type=int, help='internal at which to save a checkpoint')
@click.option('--verbose', count=True, help='Keras verbosity level')
def train_command(model_path: str, units: int, samples: int, epochs: int, validation_split: Optional[float],
                  checkpoint: int, verbose: int):
    """
    Train a model.
    """
    from keras.models import load_model

    x, y = labeled_sequences(samples, digits_with_repetition_labels)
    if Path(model_path).exists():
        model = load_model(model_path)
    else:
        model = sequence_to_sequence_model(time_steps=x.shape[1], labels=y.shape[2], units=units)
    callbacks = []
    if checkpoint:
        callbacks.append(ModelCheckpoint(filepath=model_path, verbose=verbose, save_best_only=True, period=checkpoint))
    model.fit(x, y, epochs=epochs, validation_split=validation_split, callbacks=callbacks, verbose=verbose)
    model.save(model_path)


@telos_command.command('predict')
@click.argument('model_path', metavar='model', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--test-samples', default=1000, show_default=True, help='number of samples to evaluate')
@click.option('--details', help='show prediction details')
@click.option('--verbose', count=True, help='Keras verbosity level')
def predict_command(model_path: str, test_samples: int, details: bool, verbose: int):
    """
    Evaluate a randomly generated test set.
    """
    from keras.models import load_model

    def label_row(ns: array) -> str:
        return ' '.join([' ', '^'][n] for n in ns)

    model = load_model(model_path)
    x, y = labeled_sequences(test_samples, digits_with_repetition_labels)
    y_ = model.predict(x, verbose=verbose)
    x = x[:, :, 0]
    y_true = y.argmax(axis=2)
    y_predicted = y_.argmax(axis=2)
    if details:
        for i in range(x.shape[0]):
            print(' '.join(str(n) for n in x[i]))
            print(label_row(y_true[i]))
            print(label_row(y_predicted[i]))
            print()
    accuracy = sum(y_true == y_predicted) / size(y_true)
    print(f'Accuracy: {accuracy:0.4}')


if __name__ == '__main__':
    telos_command()
