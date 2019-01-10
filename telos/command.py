from pathlib import Path
from typing import List, Optional

import click
from numpy import sum, size
from numpy.core.multiarray import array

from telos import __version__
from telos.model import sequence_to_sequence_model, digits_with_repetition_labels, labeled_sequences


@click.group('telos')
@click.version_option(__version__)
def telos_command():
    """
    Deep Learning for Long Distance Dependencies
    """
    pass


@telos_command.command('train')
@click.argument('model_path', metavar='model', type=click.Path(dir_okay=False, readable=True))
@click.option('--units', '-u', type=int, default=[16], multiple=True, show_default=True,
              help='hidden units in the model')
@click.option('--samples', default=10000, show_default=True, help='number of training samples')
@click.option('--epochs', default=200, show_default=True, help='number of training epochs')
@click.option('--validation-split', type=float, help='portion of data to use for validation')
@click.option('--checkpoint', type=int, help='internal at which to save a checkpoint')
@click.option('--verbose', '-v', count=True, help='Keras verbosity level')
def train_command(model_path: str, units: List[int], samples: int, epochs: int, validation_split: Optional[float],
                  checkpoint: int, verbose: int):
    """
    Train a model.
    """
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint

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
    model.summary()


@telos_command.command('predict')
@click.argument('model_path', metavar='model', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--test-samples', default=1000, show_default=True, help='number of samples to evaluate')
@click.option('--details', help='show prediction details')
@click.option('--verbose', '-v', count=True, help='Keras verbosity level')
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
