from setuptools import setup, find_packages

setup(
    name='needle',
    version='0.0.1',
    packages=find_packages(include=['needle', 'needle.*']),
    # other setup parameters
)