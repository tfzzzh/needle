from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
import math
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self._device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)

        out = softmax(Q @ K^T / sqrt(d)) @ V
              [B, H, T, T]                 [B, H, T, D] 

        apply mask: + mask

        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim
        assert queries_len == keys_values_len

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        # compute probs
        logits = self.matmul(q, k) # now shape: [B, H, T, T]
        logits = ops.divide_scalar(logits, math.sqrt(q_dim))

        if self.causal:
            mask = self.create_causal_mask(queries_len, queries_len, self._device) # !!! change module.device
            logits = ops.add_scalar(logits, mask.broadcast_to(logits.shape))

        probs = self.softmax(logits)
        probs = self.dropout(probs)

        Vtrans = ops.transpose(v, (2, 3))
        result = self.matmul(probs, Vtrans)
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self._device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape
        assert queries_len == keys_values_len

        ### BEGIN YOUR SOLUTION
        # perform layernorm on q, k, v
        q_mat = ops.reshape(q, (batch_size * queries_len, q_dim))
        q_mat = self.prenorm_q(q_mat)
        # q = ops.reshape(q_mat, q.shape)

        k_mat = ops.reshape(k, (batch_size * queries_len, k_dim))
        k_mat = self.prenorm_k(k_mat)
        # k = ops.reshape(k_mat, k.shape)

        v_mat = ops.reshape(v, (batch_size * queries_len, v_dim))
        v_mat = self.prenorm_v(v_mat)
        # v = ops.reshape(v_mat, v.shape)

        # project to latent space
        q_mat = self.q_projection(q_mat) # []
        k_mat = self.k_projection(k_mat)
        v_mat = self.v_projection(v_mat)         

        # reshape q, k, v to get head
        q = ops.reshape(q_mat, (batch_size, queries_len, self.num_head, self.dim_head))
        q = ops.transpose(q, (1,2))
        k = ops.reshape(k_mat, (batch_size, queries_len, self.num_head, self.dim_head))
        k = ops.transpose(k, (1,2))
        v = ops.reshape(v_mat, (batch_size, queries_len, self.num_head, self.dim_head))
        v = ops.transpose(v, (1,2))

        # attention: out shape [B H T D]
        out, att = self.attn(q, k, v) 

        # projection
        out = ops.transpose(out, (1,2))
        out = ops.reshape(out, (batch_size * queries_len, self.num_head * self.dim_head))
        out = self.out_projection(out)
        out = ops.reshape(out, (batch_size, queries_len, self.out_features))

        ### END YOUR SOLUTION

        return out


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self._device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.q_features = q_features

        self.attention = AttentionLayer(
            q_features, num_head, dim_head,
            k_features=q_features, v_features=q_features, out_features=q_features,
            dropout=dropout, causal=causal, device=device, dtype=dtype
        )
        self.drop1 = Dropout(dropout)
        self.layernorm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.drop2 = Dropout(dropout)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.drop3 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape
        assert x_dim == self.q_features

        ### BEGIN YOUR SOLUTION
        x = x + self.drop1(self.attention(x, x, x)) # [b, s, dim]
        y = ops.reshape(x, (batch_size * seq_len, x_dim))
        y = self.layernorm(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.drop2(y)
        y = self.linear2(y)
        y = self.drop3(y)
        y = ops.reshape(y, (batch_size, seq_len, x_dim))
        x = x + y
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self._device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        # position embedding
        self.pos_embed = Embedding(num_embeddings=sequence_len, embedding_dim=embedding_size, 
                                      device=device, dtype=dtype)

        # attention layers
        self.attentions = Sequential(
            *[
                TransformerLayer(
                    q_features=embedding_size, num_head=num_head, dim_head=dim_head,
                    hidden_size=hidden_size, dropout=dropout, causal=causal,
                    device=device, dtype=dtype
                )
                for _ in range(num_layers)
            ]
        ) # [batch_size, seq_len, embedding_size] -> [batch_size, seq_len, embedding_size]
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        # embed position
        batch_size, seqlen, _ = x.shape
        positions = np.tile(np.arange(seqlen), reps=(batch_size, 1))
        positions = ndarray.NDArray(positions, self._device)
        positions = Tensor(positions, device=self._device, dtype=x.dtype, requires_grad=True)
        positions = self.pos_embed(positions)
        
        x = x + positions
        x = self.attentions(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)