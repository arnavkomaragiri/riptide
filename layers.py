import numpy as np
import numba as nb
from tensor import Tensor

class Identity:
    def __init__(self):
        pass
    
    def get_grad_fn(self, x_in):
        def grad(x, psi=np.ones_like(x_in)):
            return Tensor(x, gradFn=x_in.gradFn)
        return grad

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise ValueError("Input must be of type Tensor")
        return Tensor(x.data)
    
    def __call__(self, x):
        out = self.forward(x)
        grad = self.get_grad_fn(x)
        return Tensor(out.data, gradFn=grad)

class Linear:
    def __init__(self, in_features, out_features, initialization="xavier"):
        self.in_features = in_features
        self.out_features = out_features

        if initialization == "xavier":
            self.weight = np.random.randn(out_features, in_features) / np.sqrt(in_features)
            self.bias = np.random.randn(out_features) / np.sqrt(in_features)
        elif initialization == "zeros":
            self.weight = np.zeros((out_features, in_features))
            self.bias = np.zeros(out_features)
        else:
            raise ValueError("Invalid initialization")

    # TODO: Fix the gradient function
    def get_grad_fn(self, x_in):
        def grad(x, psi=np.ones_like(x_in)):
            dW = np.dot(psi.T, x.data)
            dB = psi.sum(axis=0)

            self.weight += dW
            self.bias += dB

            return Tensor(np.dot(self.weight.T, psi), gradFn=x_in.gradFn)
        return grad
    
    def forward(self, x):
        return np.dot(self.weight, x.data) + self.bias
    
    def __call__(self, x):
        out = self.forward(x)
        grad = self.get_grad_fn(x)
        return Tensor(out, gradFn=grad)
    
class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def get_grad_fn(self, x_in):
        def grad(x, psi=np.ones_like(x_in)):
            g = np.where(x.data > 0, 1, self.alpha * np.exp(x.data))
            g = np.diag(g)
            return Tensor(np.dot(g, psi), gradFn=x_in.gradFn)
        return grad

    def forward(self, x):
        return np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1))
    
    def __call__(self, x):
        out = self.forward(x)
        grad = self.get_grad_fn(x)
        return Tensor(out, gradFn=grad)