"""Optimization module"""
import needle as ndl
from typing import List
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

def array_scatter(x, dtype):
    return ndl.array_api.array(x, dtype)


def check_params(params: List[ndl.nn.Parameter]):
    if len(params) == 0:
        raise Exception("parameter empty")
    
    for i in range(1, len(params)):
        if params[i].dtype != params[0].dtype:
            raise Exception("parameter type mismatch")
    
def check_gradtypes(params: List[ndl.nn.Parameter]):
    for i in range(0, len(params)):
        if params[i].grad is None:
            raise Exception("grad not computed")
        
        if params[i].grad.dtype != params[i].dtype:
            raise Exception("grad type and parameter type mismatch")

class SGD(Optimizer):
    '''
    update formula:
        u_{t+1} &= \beta u_t + (1-\beta) grad(\theta_t) \\
        \theta_{t+1} = \theta_t - \alpha u_{t+1}

    Parameters
    - `params` - list[Tensor]
    - `lr` (*float*) - learning rate
    - `momentum`(beta) (*float*) - momentum factor
    - `weight_decay` (*float*) - weight decay (L2 penalty)

    Here I only use NDArray to handle the updates
    '''
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        check_params(params)

        dtype = params[0].dtype
        self.lr = array_scatter(lr, dtype)
        self.momentum = array_scatter(momentum, dtype)
        self.u = {}
        self.weight_decay = array_scatter(weight_decay, dtype)

    def step(self):
        ### BEGIN YOUR SOLUTION
        check_gradtypes(self.params)
        for param in self.params:
            # compute gradient when there exist penalty
            x = param.realize_cached_data()
            g = param.grad.realize_cached_data()
            g = g + self.weight_decay * x

            # update search direction
            dir = None
            if param not in self.u:
                # dir = g dir = g do better job
                dir = (1 - self.momentum) * g # first dir is 0
            else:
                dir = (self.momentum) * self.u[param] + (1 - self.momentum) * g

            self.u[param] = dir

            # update parameters
            x -= self.lr * dir

        assert len(self.params) == len(self.u)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        check_params(params)
        dtype = params[0].dtype
        self.dtype = dtype
        self.lr = array_scatter(lr, dtype)
        self.beta1 = array_scatter(beta1, dtype)
        self.beta2 = array_scatter(beta2, dtype)
        self.eps = array_scatter(eps, dtype)
        self.weight_decay = array_scatter(weight_decay, dtype)
        self.t = 0

        self.m = {param: array_scatter(0, dtype) for param in self.params}
        self.v = {param: array_scatter(0, dtype) for param in self.params}

    def step(self):
        '''
        Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). 

        dynamic:

        u_{t+1} &= \beta_1 u_t + (1-\beta_1) grad 
        v_{t+1} &= \beta_2 v_t + (1-\beta_2) grad * grad 
        \hat{u}_{t+1} &= u_{t+1} / (1 - \beta_1^t) \text{(bias correction)} 
        \hat{v}_{t+1} &= v_{t+1} / (1 - \beta_2^t)  \text{(bias correction)}
        \theta_{t+1} &= \theta_t - \alpha \hat{u_{t+1}}/(\hat{v}_{t+1}^{1/2}+\epsilon)

        Parameters
        - `params` - iterable of parameters of type `needle.nn.Parameter` to optimize
        - `lr` (*float*) - learning rate
        - `beta1` (*float*) - coefficient used for computing running average of gradient
        - `beta2` (*float*) - coefficient used for computing running average of square of gradient
        - `eps` (*float*) - term added to the denominator to improve numerical stability
        - `weight_decay` (*float*) - weight decay (L2 penalty)
        '''
        ### BEGIN YOUR SOLUTION
        check_gradtypes(self.params)
        self.t += 1
        for param in self.params:
            # compute gradient when there exist penalty
            x = param.realize_cached_data()
            g = param.grad.realize_cached_data()
            g = g + self.weight_decay * x

            # compute momumtum and curv
            mom = running_average(self.m[param], g, self.beta1, self.dtype)
            curv = running_average(self.v[param], g * g, self.beta2, self.dtype)

            # update opt states
            self.m[param] = mom
            self.v[param] = curv            

            # compte search dir
            mom = mom / (array_scatter(1.0, self.dtype) - ndl.array_api.power(self.beta1, self.t))
            curv = curv / (array_scatter(1.0, self.dtype) - ndl.array_api.power(self.beta2, self.t))
            dir = mom / (ndl.array_api.sqrt(curv) + self.eps)

            # update parameter
            x -= self.lr * dir

            # check parameter type
            assert x.dtype == self.dtype
        ### END YOUR SOLUTION

def running_average(x_old, x_obs, beta, dtype):
    '''
    x_new = beta * x_old + (1-beta) * x_obs
    '''
    zeta = array_scatter(1.0, dtype) - beta
    x_new = beta * x_old + zeta * x_obs
    return x_new