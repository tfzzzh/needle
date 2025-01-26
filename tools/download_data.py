import urllib.request
import os


def download_penn_treebank(data_dir):
    ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
    for f in ['train.txt', 'test.txt', 'valid.txt']:
        if not os.path.exists(os.path.join('./data/ptb', f)):
            urllib.request.urlretrieve(ptb_data + f, os.path.join(data_dir, f))


def download_cifa10(data_dir):
    cifar10_data = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        urllib.request.urlretrieve(cifar10_data, os.path.join(data_dir, 'cifar-10-python.tar.gz'))
        os.system('tar -xzvf ' + os.path.join(data_dir, 'cifar-10-python.tar.gz') + ' -C ' + data_dir)
        os.system('rm ' + os.path.join(data_dir, 'cifar-10-python.tar.gz'))


if __name__ == '__main__':
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(os.path.join(data_dir, 'ptb')):
        os.makedirs(os.path.join(data_dir, 'ptb'))
        download_penn_treebank(os.path.join(data_dir, 'ptb'))

    download_cifa10(data_dir)