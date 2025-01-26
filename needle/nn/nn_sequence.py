"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, Linear


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one = init.ones(*(x.shape), device=x.device, dtype=x.dtype, requires_grad=False)
        return one / (one + ops.exp(-x))
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.activation = None
        if nonlinearity == 'tanh':
            self.activation = ops.tanh
        
        elif nonlinearity == 'relu':
            self.activation = ops.relu

        else:
            raise NotImplementedError()
        

        self.hidden_size = hidden_size

        # each layer contains a linear_mapping to input, and a mapping to h
        # self.linear_x = Linear(input_size, hidden_size, bias, device, dtype)
        # self.linear_h = Linear(hidden_size, hidden_size, bias, device, dtype)

        # initialize linear layer to U(-sqrt(k), sqrt(k))
        self.use_bias = bias
        self.W_ih = Parameter(init.rand(
            input_size, hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
            device=device, dtype=dtype, requires_grad=True
        ))
        self.W_hh = Parameter(init.rand(
            hidden_size, hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
            device=device, dtype=dtype, requires_grad=True
        ))

        if bias:
            self.bias_ih = Parameter(init.rand(
                1, hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
                device=device, dtype=dtype, requires_grad=True
            ))
            self.bias_hh = Parameter(init.rand(
                1, hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
                device=device, dtype=dtype, requires_grad=True
            ))

        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            bs = X.shape[0]
            h = init.zeros(bs, self.hidden_size, 
                            device=X.device, dtype=X.dtype, requires_grad=False)

        out = X @ self.W_ih
        if self.use_bias:
            out = out + (ops.broadcast_to(self.bias_ih, out.shape))

        out = out + h @ self.W_hh
        if self.use_bias:
            out = out + (ops.broadcast_to(self.bias_hh, out.shape))


        out = self.activation(out)
        return out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = [
            RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)
        ]

        for i in range(1, num_layers):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        inputs = ops.split(X, axis=0)

        # init loop with inputs[0]
        # hiddens = []
        # outputs = []
        # z = inputs[0]
        # for cell in self.rnn_cells:
        #     hout = cell.forward(z, h0)
        #     hiddens.append(hout)
        # outputs.append(hiddens[-1])

        # compute cell outputs from 1 to seq_len
        if h0 is not None:
            hiddens = ops.split(h0, axis=0)
        else:
            hiddens = [None] * self.num_layers

        outputs = []
        for t in range(0, seq_len):
            z = inputs[t]

            # compute cell(z, hidden)
            hiddens_next = []
            for layer, cell in enumerate(self.rnn_cells):
                hout = cell.forward(z, hiddens[layer])
                hiddens_next.append(hout)
                z = hout
            
            # update hidden states and outputs
            hiddens = hiddens_next
            outputs.append(hiddens[-1])

        # return results
        h_n = ops.stack(outputs, 0)
        h_t = ops.stack(hiddens, 0)
        assert h_n.shape == (seq_len, bs, self.hidden_size)
        assert h_t.shape == (self.num_layers, bs, self.hidden_size)

        return ops.make_tuple(h_n, h_t)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION

        self.hidden_size = hidden_size

        # initialize linear layer to U(-sqrt(k), sqrt(k))
        self.use_bias = bias
        self.W_ih = Parameter(init.rand(
            input_size, 4*hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
            device=device, dtype=dtype, requires_grad=True
        ))
        self.W_hh = Parameter(init.rand(
            hidden_size, 4*hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
            device=device, dtype=dtype, requires_grad=True
        ))

        if bias:
            self.bias_ih = Parameter(init.rand(
                1, 4*hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
                device=device, dtype=dtype, requires_grad=True
            ))
            self.bias_hh = Parameter(init.rand(
                1, 4*hidden_size, low=-np.sqrt(hidden_size), high=np.sqrt(hidden_size),
                device=device, dtype=dtype, requires_grad=True
            ))
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # when h is none fill it to (0, 0)
        batch, input_size = X.shape
        if h is None:
            h = (
                init.zeros(batch, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False),
                init.zeros(batch, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False)
            )

        hstate, cstate = h

        # compute linear(x) + linear(hstate) -> [bs, 4*self.hidden_size]
        out = X @ self.W_ih
        if self.use_bias:
            out = out + (ops.broadcast_to(self.bias_ih, out.shape))

        out = out + hstate @ self.W_hh
        if self.use_bias:
            out = out + (ops.broadcast_to(self.bias_hh, out.shape))

        # split previous out to i, f, g, o each of size [bs, hidden_size]
        out = ops.reshape(out, (batch, 4, self.hidden_size))
        outs = ops.split(out, axis=1)
        i, f, g, o = outs[0], outs[1], outs[2], outs[3]
        assert i.shape == (batch, self.hidden_size)

        # apply activations to i, f, g, o
        i = Sigmoid()(i)
        f = Sigmoid()(f)
        g = ops.tanh(g)
        o = Sigmoid()(o)

        # compute h', c'
        cnext = f * cstate + i * g
        hnext = o * ops.tanh(cnext)

        # return (h', c')
        return hnext, cnext 
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        assert input_size == self.input_size

        # handle the case where h is None
        if h is None:
            h = (
                init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False),
                init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=False)
            )
        else:
            assert h[0].shape == h[1].shape and h[0].shape == (self.num_layers, bs, self.hidden_size)

        # get hstate, cstate from h
        hstate, cstate = h
        
        # split X, hstate, cstate into sequences
        Xseq = ops.split(X, axis=0)
        hstate = ops.split(hstate, axis=0)
        cstate = ops.split(cstate, axis=0)

        # init outputs list
        outputs = []

        # for each input batch at time t, apply cell to it
        for t in range(seq_len):
            # init z to X[t]
            z = Xseq[t]
            hstate_next = []
            cstate_next = []

            for i, cell in enumerate(self.lstm_cells):
                # apply cells to z with input state (h[i], c[i]) 
                hnext, cnext = cell(z, (hstate[i], cstate[i]))

                # update z
                z = hnext

                # store computed states
                hstate_next.append(hnext)
                cstate_next.append(cnext)

            # update outputs
            outputs.append(z)

            # update hstate, cstate
            hstate = hstate_next
            cstate = cstate_next

        # return result
        outputs = ops.stack(outputs, axis=0)
        cstate = ops.stack(cstate, axis=0)
        hstate = ops.stack(hstate, axis=0)

        assert outputs.shape == (seq_len, bs, self.hidden_size)

        return outputs, (hstate, cstate)

        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x_onehot = init.one_hot(self.num_embeddings, x.cached_data, 
                                device=self.weight.device, dtype=self.weight.dtype, requires_grad=False)

        (seq_len, bs, num_embed) = x_onehot.shape
        x_onehot = ops.reshape(x_onehot, (seq_len*bs, num_embed))
        out = x_onehot @ self.weight
        out = ops.reshape(out, (seq_len, bs, self.embedding_dim))
        return out
        ### END YOUR SOLUTION