from .layers import Module
from .tensor import Tensor
import numpy as np

class MSELoss(Module):
    def __init__(self):
        pass

    def get_grad_fn(self, target: Tensor, pred: Tensor):
        def grad(self, psi=None) -> Tensor:
            # print(pred.data)
            # print(target.data)
            g = target.data - pred.data
            g /= target.shape[0]
            # print("GRADIENT: ", g.T)
            # print("loss gradient\n", g.T)
            return Tensor(-g, gradFn=pred.gradFn)
        return grad
    
    def forward(self, target: Tensor, pred: Tensor) -> Tensor:
        return Tensor((1 / target.shape[0]) * 0.5 * (target.data - pred.data).T @ (target.data - pred.data))

class CrossEntropyLoss(Module):
    def __init__(self):
        pass
    
    def get_grad_fn(self, target: Tensor, pred: Tensor):
        def ce_grad(self, psi=None) -> Tensor:
            return Tensor(-(1 / target.shape[0]) * (target.data - pred.data), gradFn=pred.gradFn)
        return ce_grad
    
    def forward(self, target: Tensor, pred: Tensor) -> Tensor:
        e = np.exp(pred.data)
        log_prob = pred.data - np.log(np.sum(e))
        i = np.argmax(target.data)
        l = -(1 / target.shape[0]) * target.data[i] * log_prob[i]
        return Tensor([l])