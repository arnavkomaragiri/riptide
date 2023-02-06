import numpy as np
import numba as nb

class Tensor:
    def __init__(self, x, requires_grad=True, dtype=np.float64, gradFn=None):
        self.data = np.array(x, dtype=dtype)
        if requires_grad:
            self.grad = None
        self.requires_grad = requires_grad
        self.gradFn = gradFn
        self.shape = self.data.shape
    
    def backward(self, psi=None):
        t = self
        if psi is None:
            psi = Tensor(np.ones_like(t.data))
        while t is not None and t.gradFn is not None:
            psi = t.gradFn(psi)
            t = psi
    
    def update_grad(self, g):
        if not self.requires_grad:
            raise RuntimeWarning("Tensor does not require gradients")
        if self.grad is None:
            self.grad = g
        else:
            self.grad = self.grad + g
    
    def zero_grad(self):
        if not self.requires_grad:
            raise RuntimeWarning("Tensor does not require gradients")
        self.grad = None