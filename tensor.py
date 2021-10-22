import numpy as np
import numba as nb

class Tensor:
    def __init__(self, x, requires_grad=True, dtype=np.float32, gradFn=None):
        self.data = np.array(x, dtype=dtype)
        if requires_grad:
            self.grad = np.zeros_like(x)
        else:
            self.grad = None
        self.requires_grad = requires_grad
        self.gradFn = gradFn
    
    def backward(self):
        t = self
        psi = np.ones_like(t.data)
        while t.gradFn is not None:
            psi = t.gradFn(t.data, psi)
     