# Needle: A Tiny Deep Learning Framework

The codes are mainly adapted from the homework of the course: [Deep Learning Systems](https://dlsyscourse.org/).

## Build the Package
### Install Dependencies
First, install mugrade
```sh
pip install git+https://github.com/locuslab/mugrade.git
```
### Install the package
Install the package in editable mode:
```
python setup.py develop
```
### Download Cifa and Ptb Dataset
```sh
python tools/download_data.py 
```
### Run Tests
To run the tests, use:
```
pytest
```

## Run Examples
```sh
cd app
python mlp_resnet.py
```