import numpy as np

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