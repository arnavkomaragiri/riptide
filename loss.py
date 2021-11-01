from layers import Module
from tensor import Tensor

class MSELoss(Module):
    def __init__(self):
        pass

    def get_grad_fn(self, target: Tensor, pred: Tensor):
        def grad(self, psi=None) -> Tensor:
            if psi == None:
                return Tensor(2 * pred.data.T, gradFn=pred.gradFn)
            return Tensor((2 * pred.data.T) @ psi, gradFn=pred.gradFn)
        return grad
    
    def forward(self, target: Tensor, pred: Tensor) -> Tensor:
        loss = (target.data - pred.data).T @ (target.data - pred.data)
        return Tensor(loss)