import numpy as np
from tensor import *

# class SGD:
#     def __init__(self, lr=0.01, minibatch_size=1):
#         self.lr = lr
#         self.minibatch_size = minibatch_size

#     def update(self, grads):
#         idx = np.random.randint(0, len(grads), self.minibatch_size)
#         return -self.lr * np.mean(grads[idx], axis=0)
    
# class SGD_Momentum:
#     def __init__(self, lr=0.01, momentum=0.9, minibatch_size=1):
#         self.lr = lr
#         self.momentum = momentum
#         self.v = None
    
#     def update(self, grads):
#         if self.v is None:
#             self.v = np.zeros_like(grads)
#         idx = np.random.randint(0, len(grads), self.minibatch_size)
#         grads = np.mean(grads[idx], axis=0)
#         self.v = self.momentum * self.v + (1 - self.momentum) * grads
#         return -self.lr * self.v

class SGD:
    def __init__(self, params, lr=0.01, minibatch_size=1):
        self.params = params
        self.lr = lr
        self.minibatch_size = minibatch_size
    
    def update_param_dict(self, params, prefix=""):
        for k, v in params.items():
            if isinstance(v, Tensor):
                grads = v.grad
                if grads is None:
                    pass
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                v.data += -self.lr * grads
            elif isinstance(v, dict):
                self.update_param_dict(v, prefix=f"{prefix}_{k}_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")
    
    def step(self):
        self.update_param_dict(self.params)

class SGD_Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9, minibatch_size=1):
        self.params = params
        self.opt_vars = {}
        self.lr = lr
        self.momentum = momentum
        self.minibatch_size = minibatch_size

        self.build_opt_vars()
    
    def build_opt_vars(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                self.opt_vars[name] = {'v': None}
            elif isinstance(v, dict):
                self.build_opt_vars(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")
    
    def update_param_dict(self, params, prefix=""):
        for k, v in params.items():
            if isinstance(v, Tensor):
                name = f"{prefix}_{k}"
                grads = v.grad
                if grads is None:
                    pass
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                 
                self.opt_vars[name]['v'] = self.momentum * self.opt_vars[name]['v'] + (1 - self.momentum) * grads
                v.data += -self.lr * self.opt_vars[name]['v']
            elif isinstance(v, dict):
                self.update_param_dict(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")

# class Adam:
#     def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, minibatch_size=1):
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.m = None
#         self.v = None
#         self.minibatch_size = minibatch_size
    
#     def update(self, grads):
#         idx = np.random.randint(0, len(grads), self.minibatch_size)
#         grads = np.mean(grads[idx], axis=0)

#         if self.m is None:
#             self.m = np.zeros_like(grads)
#         if self.v is None:
#             self.v = np.zeros_like(grads)
#         self.m = self.beta1 * self.m + (1 - self.beta1) * grads
#         self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
#         m_hat = self.m / (1 - self.beta1 ** (idx + 1))
#         v_hat = self.v / (1 - self.beta2 ** (idx + 1))
#         return -self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)