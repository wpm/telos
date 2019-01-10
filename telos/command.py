from pathlib import Path
from typing import List, Optional

import click
from h5py import File
from numpy import sum, size
from numpy.core.multiarray import array

from telos import __version__
from telos.model import sequence_to_sequence_model, digits_with_repetition_labels, LabeledSequences


class LabeledSequencesParamType(click.ParamType):
    name = 'sequences'

    def convert(self, value, _, __):

        try:
            with File(value) as f:
                x = array(f['x'])
                y = array(f['y'])
        except (KeyError, OSError) as e:
            self.fail(e)
        return LabeledSequences(x, y)


@click.group('telos')
@click.version_option(__version__)
def telos_group():
    """
    Deep Learning for Long Distance Dependencies
    """
    pass


@telos_group.group('data')
def data_group():
    """
    Generate and view data sets.
    """
    pass


@data_group.command('generate')
@click.argument('n', type=int)
@click.argument('output', type=click.Path(exists=False, dir_okay=False, readable=True))
def generate_command(n: int, output: str):
    """
    Generate labeled data set.
    """
    data = LabeledSequences.from_sampler(n, digits_with_repetition_labels)
    print(data)
    data.save(output)


@data_group.command('dimensions')
@click.argument('data', type=LabeledSequencesParamType())
def dimensions_command(data: LabeledSequences):
    """
    Print labeled data set dimensions.
    """
    print(data)


@telos_group.command('train')
@click.argument('model_path', metavar='MODEL', type=click.Path(dir_okay=False, writable=True))
@click.argument('data', type=LabeledSequencesParamType())
@click.option('--units', '-u', type=int, default=[16], multiple=True, show_default=True,
              help='hidden units in the model')
@click.option('--epochs', default=200, show_default=True, help='number of training epochs')
@click.option('--validation-split', type=float, help='portion of data to use for validation')
@click.option('--checkpoint', type=int, help='interval at which to save a checkpoint')
@click.option('--verbose', '-v', count=True, help='Keras verbosity level')
def train_command(model_path: str, data: LabeledSequences, units: List[int],
                  epochs: int, validation_split: Optional[float],
                  checkpoint: int, verbose: int):
    """
    Train a model.
    """
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint

    if Path(model_path).exists():
        model = load_model(model_path)
    else:
        model = sequence_to_sequence_model(data.time_steps, data.features, data.labels, units)
    callbacks = []
    if checkpoint:
        callbacks.append(ModelCheckpoint(filepath=model_path, verbose=verbose, save_best_only=True, period=checkpoint))
    model.fit(data.x, data.y, epochs=epochs, validation_split=validation_split, callbacks=callbacks, verbose=verbose)
    model.save(model_path)
    model.summary()


@telos_group.command('predict')
@click.argument('model_path', metavar='MODEL', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('data', type=LabeledSequencesParamType())
@click.option('--details', is_flag=True, help='show prediction details')
@click.option('--verbose', '-v', count=True, help='Keras verbosity level')
def predict_command(model_path: str, data: LabeledSequences, details: bool, verbose: int):
    """
    Evaluate a randomly generated test set.
    """
    from keras.models import load_model

    def label_row(ns: array) -> str:
        return ' '.join([' ', '^'][n] for n in ns)

    model = load_model(model_path)
    y_ = model.predict(data.x, verbose=verbose)
    x = data.x[:, :, 0]
    y_true = data.y.argmax(axis=2)
    y_predicted = y_.argmax(axis=2)
    if details:
        for i in range(x.shape[0]):
            print(' '.join(str(n) for n in x[i]))
            print(label_row(y_true[i]))
            print(label_row(y_predicted[i]))
            print()
    accuracy = sum(y_true == y_predicted) / size(y_true)
    print(f'Accuracy: {accuracy:0.4}')
