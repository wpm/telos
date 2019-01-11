from pathlib import Path
from typing import List, Optional

import click

from telos import __version__
from telos.model import sequence_to_sequence_model, digits_with_repetition_labels, LabeledSequences


class LabeledSequencesParamType(click.ParamType):
    name = 'sequences'

    def convert(self, value, _, __):
        try:
            return LabeledSequences.from_file(value)
        except (KeyError, OSError) as e:
            self.fail(e)


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


@data_group.command('info')
@click.argument('data', type=LabeledSequencesParamType())
@click.option('--details', is_flag=True, help='show sequence details')
def info_command(data: LabeledSequences, details: bool):
    """
    Info about a labeled data set.
    """
    if details:
        data.sequence_details()
    print(data)


@telos_group.command('train')
@click.argument('model_path', metavar='MODEL', type=click.Path(dir_okay=False, writable=True))
@click.argument('data', type=LabeledSequencesParamType())
@click.option('--units', '-u', type=int, default=[16], multiple=True, show_default=True,
              help='hidden units in the model')
@click.option('--epochs', default=200, show_default=True, help='number of training epochs')
@click.option('--validation-split', type=float, help='portion of data to use for validation')
@click.option('--checkpoint', type=int, help='interval at which to save a checkpoint')
@click.option('--tensorboard', type=click.Path(exists=False, file_okay=False, writable=True),
              help='directory in which to write TensorBoard logs')
@click.option('--limit', type=int, help='limit training set to this many samples')
@click.option('--verbose', '-v', count=True, help='verbosity level')
def train_command(model_path: str, data: LabeledSequences, units: List[int],
                  epochs: int, validation_split: Optional[float],
                  checkpoint: int, tensorboard: Optional[str], limit: Optional[int], verbose: int):
    """
    Train a model.
    """
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint, TensorBoard

    if Path(model_path).exists():
        model = load_model(model_path)
    else:
        model = sequence_to_sequence_model(data.time_steps, data.features, data.labels, units)
    callbacks = []
    if checkpoint:
        callbacks.append(ModelCheckpoint(filepath=model_path, verbose=verbose, save_best_only=True, period=checkpoint))
    if tensorboard:
        callbacks.append(TensorBoard(tensorboard))
    data = data.limit(limit)
    print(data)
    model.fit(data.x, data.y, epochs=epochs, validation_split=validation_split, callbacks=callbacks,
              verbose=keras_verbosity(verbose, True))
    model.save(model_path)
    model.summary()


@telos_group.command('predict')
@click.argument('model_path', metavar='MODEL', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('data', type=LabeledSequencesParamType())
@click.option('--details', is_flag=True, help='show prediction details')
@click.option('--verbose', '-v', count=True, help='verbosity level')
def predict_command(model_path: str, data: LabeledSequences, details: bool, verbose: int):
    """
    Evaluate a test set.
    """
    from keras.models import load_model

    model = load_model(model_path)
    y = model.predict(data.x, verbose=keras_verbosity(verbose, False))
    print(data)
    if details:
        data.sequence_details(y)
    print(f'Accuracy: {data.accuracy(y):0.4}')


def keras_verbosity(verbose: int, train: bool):
    if train:
        if verbose == 1:
            verbose = 2
        elif verbose >= 2:
            verbose = 1
    else:
        verbose = min(verbose, 1)
    return verbose
