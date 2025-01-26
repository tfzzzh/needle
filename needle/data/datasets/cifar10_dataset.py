import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.train = train
        self.p = p
        self.transforms = transforms

        if self.train:
            self.images, self.labels = parse_cifa10_train(base_folder, normalize=True)
        else:
            self.images, self.labels = parse_cifa10_test(base_folder, normalize=True)
        
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index, int) or isinstance(index, np.integer):
            x, y = self.images[index], self.labels[index]
            assert x.shape == (3, 32, 32)

            if self.transforms is not None:
                for transform in self.transforms:
                    x = transform(x)
        
            return x, y
        
        elif isinstance(index, slice):
            x, y = self.images[index], self.labels[index]
            n = len(x)

            if self.transforms is not None:
                for i in range(n):
                    for transform in self.transforms:
                        x[i] = transform(x[i])
            
            return x, y
        
        else:
            raise NotImplementedError
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION


def parse_cifa10_train(data_dir, normalize=True):
    '''
    return:
        X: nd.array of shape 50000 * 3 * 32 * 32

        y: labels -- a list of 50000 numbers in the range 0-9. The number at index i 
            indicates the label of the ith image in the array data.
    '''
    Xs = []
    ys = []
    for i in range(5):
        file_name = os.path.join(data_dir, f'data_batch_{i+1}')
        x, y =  parse_cifa10_batch(file_name, normalize)
        Xs.append(x)
        ys.append(y)

    return np.concatenate(Xs, axis=0), np.concatenate(ys)


def parse_cifa10_test(data_dir, normalize=True):
    '''
    return:
        X: nd.array of shape 10000 * 3 * 32 * 32

        y: labels -- a list of 10000 numbers in the range 0-9. The number at index i 
            indicates the label of the ith image in the array data.
    '''
    file_name = os.path.join(data_dir, 'test_batch')
    return parse_cifa10_batch(file_name, normalize)


def parse_cifa10_batch(file_name, normalize):
    with open(file_name, 'rb') as fo:
        # Each row of the array stores a 32x32 colour image 
        dictionary = pickle.load(fo, encoding='bytes')
        X = dictionary[b'data'].reshape(10000, 3, 32, 32)
        if not normalize:
            X = X.astype('float32')
        else:
            X = (X / 255).astype('float32')
        y = np.array(dictionary[b'labels'], dtype='uint8')

        return X, y