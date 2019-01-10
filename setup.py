from setuptools import setup, find_packages

from telos.version import version


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='Telos',
    version=version,
    packages=find_packages(),
    python_requires='>=3.6',
    url='https://github.com/wpm/telos',
    author='W.P. McNeill',
    author_email='billmcn@gmail.com',
    description='Deep Learning for Long Distance Dependencies',
    long_description=readme(),
    entry_points={'console_scripts': ['telos=telos.command:telos_group']},
    install_requires=['click', 'numpy', 'keras', 'h5py']
)
