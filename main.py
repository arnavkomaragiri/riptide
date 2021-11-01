from layers import *
from tensor import Tensor
from loss import *

import numpy as np

class LogisticRegression(Module):
    def __init__(self, in_features, out_features):
        self.linear = Linear(in_features, out_features)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = LogisticRegression(2, 1)
input = Tensor(np.ones(2))
output = model(input)

loss_fn = MSELoss()
loss = loss_fn(Tensor([1]), output)
model.zero_grad()
loss.backward()

# TODO: Implement Logistic Regression on the MNIST dataset