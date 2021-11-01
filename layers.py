import numpy as np
import numba as nb
from abc import abstractmethod
from tensor import Tensor

class Module:
    def __init__(self):
        pass

    @abstractmethod
    def get_grad_fn(self, *inputs):
        def grad(psi=None) -> Tensor:
            out = self.forward(*inputs)
            return out.backward(psi=psi)
        return grad

    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        pass
    
    def __call__(self, *inputs):
        out = self.forward(*inputs)
        grad = self.get_grad_fn(*inputs)
        return Tensor(out.data, gradFn=grad)
    
    def zero_grad(self):
        vars = [v for v in self.__dict__.values() if isinstance(v, Module) or isinstance(v, Tensor)]
        for var in vars:
            var.zero_grad()

    def parameters(self):
        vars = [(n, v) for n, v in self.__dict__.items() if isinstance(v, Module) or isinstance(v, Tensor)]
        return {n: v.parameters() if isinstance(v, Module) else v for n, v in vars}

class Identity(Module):
    def __init__(self):
        pass
    
    def get_grad_fn(self, x_in: Tensor):
        # this is prob wrong but I really don't care much
        def identity_grad(psi=Tensor(np.ones_like(x_in))) -> Tensor:
            return Tensor(x_in.data, gradFn=x_in.gradFn)
        return identity_grad

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise ValueError("Input must be of type Tensor")
        return Tensor(x.data)

class Linear(Module):
    def __init__(self, in_features, out_features, initialization="xavier"):
        self.in_features = in_features
        self.out_features = out_features

        if initialization == "xavier":
            self.weight = Tensor(np.random.randn(out_features, in_features) / np.sqrt(in_features))
            self.bias = Tensor(np.random.randn(out_features) / np.sqrt(in_features))
        elif initialization == "zeros":
            self.weight = Tensor(np.zeros((out_features, in_features)))
            self.bias = Tensor(np.zeros(out_features))
        else:
            raise ValueError("Invalid initialization")

    def get_grad_fn(self, x_in: Tensor):
        def linear_grad(psi=Tensor(np.ones_like(x_in))) -> Tensor:
            # this makes me want to vomit
            if self.out_features == 1:
                dW = psi.data * x_in.data
            else:
                dW = psi.data.T @ x_in.data
            dB = psi.data.sum(axis=0)

            self.weight.update_grad(dW)
            self.bias.update_grad(dB)

            return Tensor(self.weight.data.T @ psi.data, gradFn=x_in.gradFn)
        return linear_grad

    # TODO: Add safety checks for batching 
    def forward(self, x: Tensor) -> Tensor:
        return self.weight.data @ x.data + self.bias.data
    
class ELU(Module):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def get_grad_fn(self, x_in: Tensor):
        def elu_grad(psi=Tensor(np.ones_like(x_in))) -> Tensor:
            g = np.where(x_in.data > 0, 1, self.alpha * np.exp(x_in.data))
            g = np.diag(g)
            return Tensor(np.dot(g, psi), gradFn=x_in.gradFn)
        return elu_grad

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1)))

class Sigmoid(Module):
    def __init__(self):
        self.out = None

    def get_grad_fn(self, x_in: Tensor):
        def sigmoid_grad(psi=Tensor(np.ones_like(x_in))) -> Tensor:
            if self.out is not None:
                return Tensor(np.diag(self.out.data * (1 - self.out.data)), gradFn=x_in.gradFn)
            return Tensor(np.exp(-x_in.data) / np.square(1 + np.exp(-x_in.data)), gradFn=x_in.gradFn)
        return sigmoid_grad
    
    def forward(self, x: Tensor) -> Tensor:
        self.out = Tensor(1 / (1 + np.exp(-x.data)))
        return self.out