from setuptools import setup, find_packages

setup(
    name='FedDifferentialPrivacyTree',
    version='1.0.0',
    author='Hannes Gubler',
    author_email='hannes.gubler@epfl.ch',
    description='This package implements a federated differentially private decision tree.',
    packages=["FedDifferentialPrivacyTree"],
    install_requires=[
        'numpy'
    ],
    #classifiers=[
    #    'Programming Language :: Python :: 3'
    #],
)
