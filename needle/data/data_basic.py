import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        
        self.batch_idx = 0
        assert(self.batch_size > 0)

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            self.ordering = np.array_split(
                np.random.permutation(len(self.dataset)), 
                range(self.batch_size, len(self.dataset), self.batch_size))
        self.batch_idx = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_idx >= len(self.ordering):
            raise StopIteration

        # x_batch = []
        # y_batch = []
        data = None
        indices = self.ordering[self.batch_idx]
        for idx in indices:
            point = self.dataset[idx]
            if data is None:
                data = []
                for _ in range(len(point)):
                    data.append([])
            
            else:
                assert len(data) == len(point)

            # for most case:
            # x = np.expand_dims(x, axis=0)
            # x_batch.append(x)
            # y_batch.append(y)
            for i, x in enumerate(point):
                # assumption: x is of type NDArray
                # assert isinstance(x, np.ndarray), f"{x} is of type {type(x)} not ndarray"
                if not isinstance(x, np.ndarray):
                    x = np.array(x)

                # for x of shape (d1,d2..) make the first dimension being batch_size
                if (len(x.shape) > 0):
                    x = np.expand_dims(x, axis=0)
                
                data[i].append(x)
        
        # x_batch = Tensor(np.concatenate(x_batch, axis=0), requires_grad=False)
        # y_batch = Tensor(np.array(y_batch), requires_grad=False)
        for i in range(len(data)):
            if len(data[i][0].shape) > 0:
                data[i] = Tensor(np.concatenate(data[i]), requires_grad=False)
            else:
                data[i] = Tensor(np.array(data[i]), requires_grad=False)

        self.batch_idx += 1
        return tuple(data)
        ### END YOUR SOLUTION

