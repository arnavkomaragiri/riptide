import numpy as np
from .tensor import *

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
                    continue
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                else:
                    grads /= self.minibatch_size
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

        self.build_opt_vars(self.params)
    
    def build_opt_vars(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                self.opt_vars[name] = {'v': np.zeros_like(v.data)}
            elif isinstance(v, dict):
                self.build_opt_vars(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")
    
    def update_param_dict(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                grads = v.grad
                if grads is None:
                    continue
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                else:
                    grads /= self.minibatch_size
                 
                # print(name+"\n", self.opt_vars[name]['v'])
                self.opt_vars[name]['v'] = self.momentum * self.opt_vars[name]['v'] + (1 - self.momentum) * grads
                # print(name+"\n", self.opt_vars[name]['v'])
                # print(v.data)
                # print(grads)
                v.data += -self.lr * self.opt_vars[name]['v']
            elif isinstance(v, dict):
                self.update_param_dict(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")

    def step(self):
        self.update_param_dict(self.params)

class SGD_Demon:
    def __init__(self, params, iters, lr=0.01, momentum=0.9, minibatch_size=1):
        self.params = params
        self.iters = iters
        self.lr = lr
        self.momentum = momentum
        self.minibatch_size = minibatch_size
        self.opt_vars = {}

        self.build_opt_vars(params)

    def build_opt_vars(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                self.opt_vars[name] = {'v': np.zeros_like(v.data), 't': 0}
            elif isinstance(v, dict):
                self.build_opt_vars(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")

    def update_param_dict(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                grads = v.grad
                if grads is None:
                    continue
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                else:
                    grads /= self.minibatch_size

                t = self.opt_vars[name]['t']

                # Taken from https://arxiv.org/pdf/1910.04952.pdf
                p_t = (self.iters - t) / self.iters
                momentum = self.momentum * (p_t / (1 - self.momentum + (self.momentum * p_t)))

                self.opt_vars[name]['t'] = t + 1
                self.opt_vars[name]['v'] = momentum * self.opt_vars[name]['v'] + (1 - momentum) * grads

                dw = -self.lr * self.opt_vars[name]['v']
                # print(np.max(dw))

                v.data += -self.lr * self.opt_vars[name]['v']
            elif isinstance(v, dict):
                self.update_param_dict(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")

    def step(self):
        self.update_param_dict(self.params)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, minibatch_size=1, epsilon=1e-8):
        self.params = params
        self.opt_vars = {}
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon

        self.build_opt_vars(self.params)
    
    def build_opt_vars(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                self.opt_vars[name] = {'v': np.zeros_like(v.data), 'm': np.zeros_like(v.data), 't': 0}
            elif isinstance(v, dict):
                self.build_opt_vars(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")

    def update_param_dict(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                grads = v.grad
                if grads is None:
                    continue
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                else:
                    grads /= self.minibatch_size
                 
                m, vo, t = self.opt_vars[name]['m'], self.opt_vars[name]['v'], self.opt_vars[name]['t']

                m = self.beta1 * m + (1 - self.beta1) * grads
                vo = self.beta2 * vo + (1 - self.beta2) * grads ** 2
                t += 1

                self.opt_vars[name]['m'] = m
                self.opt_vars[name]['v'] = vo
                self.opt_vars[name]['t'] = t

                m_hat = m / (1 - self.beta1 ** t)
                v_hat = vo / (1 - self.beta2 ** t)

                v.data += -self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)
            elif isinstance(v, dict):
                self.update_param_dict(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")
    
    def step(self):
        self.update_param_dict(self.params)

class Adamax:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, minibatch_size=1, epsilon=1e-8):
        self.params = params
        self.opt_vars = {}
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon

        self.build_opt_vars(self.params)
    
    def build_opt_vars(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                self.opt_vars[name] = {'u': np.zeros_like(v.data), 'm': np.zeros_like(v.data), 't': 0}
            elif isinstance(v, dict):
                self.build_opt_vars(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")

    def update_param_dict(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                grads = v.grad
                if grads is None:
                    continue
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                else:
                    grads /= self.minibatch_size
                 
                m, u, t = self.opt_vars[name]['m'], self.opt_vars[name]['u'], self.opt_vars[name]['t']

                m = self.beta1 * m + (1 - self.beta1) * grads
                u = np.maximum(self.beta2 * u, np.abs(grads))
                # print(u)
                # print(np.abs(grads))
                # input()
                t += 1

                self.opt_vars[name]['m'] = m
                self.opt_vars[name]['u'] = u 
                self.opt_vars[name]['t'] = t

                scale = self.lr / (1 - (self.beta1 ** t))
                v.data += -scale * m / (u + self.epsilon)
            elif isinstance(v, dict):
                self.update_param_dict(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")
    
    def step(self):
        self.update_param_dict(self.params)
class Adam_Demon:
    def __init__(self, params, iters, lr=0.001, beta1=0.9, beta2=0.999, minibatch_size=1, epsilon=1e-8):
        self.params = params
        self.iters = iters
        self.opt_vars = {}
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon

        self.build_opt_vars(self.params)
    
    def build_opt_vars(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                self.opt_vars[name] = {'v': np.zeros_like(v.data), 'm': np.zeros_like(v.data), 't': 0}
            elif isinstance(v, dict):
                self.build_opt_vars(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")

    def update_param_dict(self, params, prefix=""):
        for k, v in params.items():
            name = f"{prefix}_{k}"
            if isinstance(v, Tensor):
                grads = v.grad
                if grads is None:
                    continue
                if len(grads.shape) != len(v.shape):
                    if self.minibatch_size > grads.shape[0]:
                        raise ValueError(f"Minibatch Size of {self.minibatch_size} is greater than gradient size of {grads.shape[0]}")
                    idx = np.random.choice(range(grads.shape[0]), self.minibatch_size, replace=False)
                    grads = np.mean(grads[idx], axis=0)
                else:
                    grads /= self.minibatch_size
                 
                m, vo, t = self.opt_vars[name]['m'], self.opt_vars[name]['v'], self.opt_vars[name]['t']

                # Taken from https://arxiv.org/pdf/1910.04952.pdf
                p_t = (self.iters - t) / self.iters
                beta_t = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))

                m = self.beta1 * m + (1 - beta_t) * grads
                vo = self.beta2 * vo + (1 - self.beta2) * grads ** 2
                t += 1

                self.opt_vars[name]['m'] = m
                self.opt_vars[name]['v'] = vo
                self.opt_vars[name]['t'] = t

                m_hat = m / (1 - self.beta1 ** t)
                v_hat = vo / (1 - self.beta2 ** t)

                v.data += -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            elif isinstance(v, dict):
                self.update_param_dict(v, prefix=name+"_")
            else:
                raise RuntimeWarning(f"Encountered Unknown Parameter Type: {k}: {v}")
    
    def step(self):
        self.update_param_dict(self.params)