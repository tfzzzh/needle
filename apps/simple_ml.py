"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import needle as ndl
import needle.nn as nn
from models import *
import time


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


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    n, nc = Z.shape
    
    # log sum exp(z_i)
    logsum = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))

    # get logits at y
    Zy = ndl.summation(Z * y_one_hot, axes=(1,))

    return ndl.summation(logsum - Zy) / np.array(n, dtype=Z.dtype)
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    n = X.shape[0]
    num_class = W2.shape[1]

    # implement net forward lambda
    model = lambda X: ndl.matmul(
        ndl.relu(
            ndl.matmul(
                X, W1
            )
        ),
        W2
    )


    for i in range(0, n, batch):
        start_idx = i
        end_idx = min(n, i + batch)

        Xbatch = X[start_idx : end_idx]
        ybatch = y[start_idx : end_idx]
        ybatch = onehot(ybatch, num_class, np.uint8)

        # move data to tensor
        Xbatch = ndl.Tensor(Xbatch, requires_grad=False)
        ybatch = ndl.Tensor(ybatch, requires_grad=False)

        # forward
        logits = model(Xbatch)
        loss = softmax_loss(logits, ybatch)

        # backward
        loss.backward()

        # optimize
        W1.data -= lr * W1.grad.data
        W2.data -= lr * W2.grad.data

    return W1, W2

    ### END YOUR SOLUTION

# implement onehot
def onehot(y, num_class, dtype):
    n = len(y)
    Y = np.zeros(shape=(n, num_class), dtype=dtype)
    Y[np.arange(n), y] = 1
    return Y

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)


### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    # initialize aggregation variables
    loss_agg = 0.0
    error_agg = 0.0
    tot_entries = 0

    # iterate over batches from the dataloader
    idx = 0
    for (x_batch, y_batch) in dataloader:
        # if not len(x_batch.shape) == 2:
        #     bsize = x_batch.shape[0]
        #     x_batch = ndl.ops.reshape(x_batch, (bsize, 28*28))

        # assert len(y_batch.shape) == 1
        model_device = model.device
        if model_device != x_batch.device:
            x_batch = ndl.Tensor(x_batch.cached_data, device=model_device, dtype=x_batch.dtype)
            y_batch = ndl.Tensor(y_batch.cached_data, device=model_device, dtype=y_batch.dtype)

        # forward pass compute logits
        logits = model(x_batch)

        # compute current loss
        loss = loss_fn(logits, y_batch)

        # optimize when opt not none
        if opt is not None:
            loss.backward()
            opt.step()
            opt.reset_grad()

        # aggregate error and loss
        num_entries = x_batch.shape[0]
        tot_entries += num_entries
        loss_agg += num_entries * loss.realize_cached_data().numpy()
        error_agg += count_error(logits, y_batch)

        idx += 1
        if idx % 100 == 0:
            print("\t consume %d batches" % idx)

    # return average error and loss (in float)
    return error_agg/ tot_entries, loss_agg / tot_entries
    ### END YOUR SOLUTION

def count_error(logits: nn.Tensor, y_true: nn.Tensor) -> np.integer:
    logits_data = logits.realize_cached_data().numpy()
    y_data = y_true.realize_cached_data().numpy().astype('int64')

    # check y_data is integer
    assert np.issubdtype(y_data.dtype, np.integer), f"y_data must be an integer array, but is of type {y_data.dtype}"

    # get the predicted class
    pred = np.argmax(logits_data, axis=1)

    return np.sum(pred != y_data)


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss = loss_fn()
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # foreach epoch train the model and print train loss and train error
    for i in range(n_epochs):
        train_error, train_loss = epoch_general_cifar10(dataloader, model, loss, opt)
        print(f"epoch = {i} \t train_error={train_error}, train_loss={train_loss}")
    
    return train_error, train_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss = loss_fn()
    valid_error, valid_loss = epoch_general_cifar10(dataloader, model, loss, opt=None)
    return valid_error, valid_loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    # initialize aggregation variables
    loss_agg = 0.0
    error_agg = 0.0
    tot_entries = 0

    # for each data batch collect loss and accurcy
    num_batch = data.shape[0] - seq_len
    for idx in range(num_batch):
        # xbatch: (seq_len, batch_size), ybatch: (seq_len * batch_size)
        xbatch, ybatch = ndl.data.datasets.ptb_dataset.get_batch(
            data, idx, seq_len, device, dtype
        )

        # forward to logits
        logits, _ = model(xbatch)

        # compute loss
        loss = loss_fn(logits, ybatch)

       # optimize when opt not none
        if opt is not None:
            loss.backward()
            opt.step()
            opt.reset_grad()

        # aggregate error and loss
        num_entries = ybatch.shape[0]
        tot_entries += num_entries
        loss_agg += num_entries * loss.realize_cached_data().numpy()
        error_agg += count_error(logits, ybatch)

        if (idx+1) % 1000 == 0:
            print("\t consume %d batches" % idx)

    return error_agg/ tot_entries, loss_agg / tot_entries
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss = loss_fn()
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(n_epochs):
        train_error, train_loss = epoch_general_ptb(data, model, seq_len, loss, opt, clip, device, dtype)
        print(f"epoch = {i} \t train_error={train_error}, train_loss={train_loss}")
    
    return train_error, train_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss = loss_fn()
    valid_error, valid_loss = epoch_general_ptb(data, model, seq_len, loss, None, None, device, dtype)
    
    return valid_error, valid_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


# def loss_err(h, y):
#     """Helper function to compute both loss and error"""
#     y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
#     y_one_hot[np.arange(y.size), y] = 1
#     y_ = ndl.Tensor(y_one_hot)
#     return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
