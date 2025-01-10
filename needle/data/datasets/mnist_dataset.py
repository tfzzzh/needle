from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.images = None
        self.labels = None
        self.transforms = transforms

        self.images, self.labels = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, int) or isinstance(index, np.integer):
            x, y = self.images[index], self.labels[index]
            assert len(x) == 28 * 28, f'at {index} the image has length {len(x)} not equal to {28 * 28}'
            x = np.reshape(x, (28, 28, 1))

            if self.transforms is not None:
                for transform in self.transforms:
                    x = transform(x)
        
            return x, y
        
        elif isinstance(index, slice):
            x, y = self.images[index], self.labels[index]
            n = len(x)
            x = np.reshape(x, (n, 28, 28, 1))

            if self.transforms is not None:
                for i in range(n):
                    for transform in self.transforms:
                        x[i] = transform(x[i])
            
            return x, y
        else:
            raise NotImplementedError

        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    images = parse_mnist_image(image_filesname)
    labels = parse_mnist_label(label_filename)
    return images, labels
    ### END YOUR CODE

def read_struct(f, fmt):
    struct_len = struct.calcsize(fmt)
    struct_unpack = struct.Struct(fmt).unpack_from
    data = f.read(struct_len)
    
    if data is None: return None

    return struct_unpack(data)


def parse_mnist_label(label_filename):
    '''
    # Parsing Mnist
    big endian
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    '''
    labels = []
    with gzip.open(label_filename, 'rb') as file:
        magic = read_struct(file, '>i')[0]
        assert magic == 2049
        len = read_struct(file, '>i')[0]
        
        for i in range(len):
            labels.append(read_struct(file, '>b')[0])
        
    return np.array(labels, dtype=np.uint8)


def parse_mnist_image(image_filename):
    '''
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 
        255 means foreground (black). 
    '''
    images = []
    with gzip.open(image_filename, 'rb') as file:
        magic = read_struct(file, '>i')[0]
        assert magic == 2051
        len = read_struct(file, '>i')[0]
        num_row = read_struct(file, '>i')[0]
        num_col = read_struct(file, '>i')[0]

        for i in range(len):
            image = read_struct(file, f'{num_row * num_col}B')
            images.append(image)
    
    images = np.array(images, dtype=np.uint8)
    images = np.array(images, dtype=np.float32) / 255
    return images