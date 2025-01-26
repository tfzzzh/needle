import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1) -> nn.Module:
    '''

    # Parameters:
    - `dim` (*int*) - input dim
    - `hidden_dim` (*int*) - hidden dim
    - `norm` (*nn.Module*) - normalization method
    - `drop_prob` (*float*) - dropout probability
    '''
    ### BEGIN YOUR SOLUTION
    # structure:
    # linear layer dim -> hidden_dim
    # batchnorm: hidden_dim -> hidden_dim
    # relu
    # dropout
    # linear
    # norm
    # skip connection
    # ReLu
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    rb = nn.Sequential(
        nn.Residual(fn),
        nn.ReLU()
    )
    return rb
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    '''
    structrue
    Linear: dim -> hidden
    ResidualBlock: hidden -> hidden / 2 -> hidden
    ...
    Linear: hidden -> class
    '''
    ### BEGIN YOUR SOLUTION
    net = [nn.Linear(dim, hidden_dim), nn.ReLU()]

    for i in range(num_blocks):
        net.append(
            ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob)
        )

    net.append(
        nn.Linear(hidden_dim, num_classes)
    )

    return nn.Sequential(*net)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    ''' one epoch of training or evaluation
    iterating over the entire training dataset once, 
    Returns the average error rate (as a float) and the average loss over all samples (as a float)

    ##### Parameters
    - `dataloader` (*`needle.data.DataLoader`*) - dataloader returning samples from the training dataset
    - `model` (*`needle.nn.Module`*) - neural network
    - `opt` (*`needle.optim.Optimizer`*) - optimizer instance, or `None`

    '''
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # set model to train of eval according to opt
    if opt is not None:
        model.train()
    else:
        model.eval()

    # init lossfn to cross entropy
    loss_fn = nn.SoftmaxLoss()

    # initialize aggregation variables
    loss_agg = 0.0
    error_agg = 0.0
    tot_entries = 0

    # iterate over batches from the dataloader
    for (x_batch, y_batch) in dataloader:
        if not len(x_batch.shape) == 2:
            bsize = x_batch.shape[0]
            x_batch = ndl.ops.reshape(x_batch, (bsize, 28*28))

        assert len(y_batch.shape) == 1

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
        loss_agg += num_entries * loss.realize_cached_data()
        error_agg += count_error(logits, y_batch)

    # return average error and loss (in float)
    return error_agg/ tot_entries, loss_agg / tot_entries
    ### END YOUR SOLUTION

# I use numpy must change it to array_api when cuda enabled
def count_error(logits: nn.Tensor, y_true: nn.Tensor) -> np.integer:
    logits_data = logits.realize_cached_data().numpy()
    y_data = y_true.realize_cached_data().numpy()

    # check y_data is integer
    if not np.issubdtype(y_data.dtype, np.integer):
        y_data = y_data.astype('int64')

    # get the predicted class
    pred = np.argmax(logits_data, axis=1)

    return np.sum(pred != y_data)


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    '''
    Initializes a training dataloader (with `shuffle` set to `True`) and a test dataloader for MNIST data, 
    and trains an `MLPResNet` using the given optimizer (if `opt` is not None) and the softmax loss for 
    a given number of epochs. Returns a tuple of the training error, training loss, test error, test loss 
    computed in the last epoch of training. If any parameters are not specified, use the default parameters.
    '''
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # load data using dataloader
    train_dataset = ndl.data.MNISTDataset(
        image_filename=data_dir + '/' + 'train-images-idx3-ubyte.gz', 
        label_filename=data_dir + '/' + 'train-labels-idx1-ubyte.gz',
        transforms = [lambda x: np.reshape(x, (-1,))]
    )

    test_dataset = ndl.data.MNISTDataset(
        image_filename=data_dir + '/' + 't10k-images-idx3-ubyte.gz', 
        label_filename=data_dir + '/' + 't10k-labels-idx1-ubyte.gz',
        transforms = [lambda x: np.reshape(x, (-1,))]
    )

    train_loader = ndl.data.DataLoader(train_dataset, batch_size, True)
    test_loader = ndl.data.DataLoader(test_dataset, batch_size, False)

    # build the model
    model = MLPResNet(dim=28*28, hidden_dim=hidden_dim)

    # set optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # foreach epoch train the model and print train loss and train error
    for i in range(epochs):
        train_error, train_loss = epoch(train_loader, model, opt=opt)
        print(f"epoch = {i} \t train_error={train_error}, train_loss={train_loss}")

    # when training finish report test error
    test_error, test_loss = epoch(test_loader, model, opt=None)

    print(f"train complete with test_error={test_error}, test_loss={test_loss}")

    # return training error, training loss, test error, test loss
    ### END YOUR SOLUTION
    return train_error, train_loss, test_error, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
