from os import path
from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.rst')) as f:
    readme = f.read()

setup(
    name             = 'lld',
    version          = '0.2',
    description      = 'An inference app to use pre-trained weights to measure length discrepancy in input leg images',
    long_description = readme,
    author           = 'FNNDSC',
    author_email     = 'dev@babyMRI.org',
    url              = 'http://wiki',
    packages         = ['lld',
                        'LLDcode',
                        'LLDcode.datasets',
                        'LLDcode.graph',
                        'LLDcode.datasources',
                        'LLDcode.generators',
                        'LLDcode.iterators',
                        'LLDcode.tensorflow_train',
                        'LLDcode.tensorflow_train.utils',
                        'LLDcode.tensorflow_train.layers',
                        'LLDcode.tensorflow_train.losses',
                        'LLDcode.tensorflow_train.networks',
                        'LLDcode.utils',
                        'LLDcode.utils.io',
                        'LLDcode.utils.landmark',
                        'LLDcode.transformations',
                        'LLDcode.transformations.spatial',
                        'LLDcode.transformations.intensity.np',
                        'LLDcode.transformations.intensity.sitk'],
    install_requires = ['chrisapp'],
    test_suite       = 'nose.collector',
    tests_require    = ['nose'],
    license          = 'MIT',
    zip_safe         = False,
    entry_points     = {
        'console_scripts': [
            'lld = lld.__main__:main'
            ]
        }
)
