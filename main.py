from .layers import *
from .tensor import Tensor
from .loss import *
from .optim import *

import numpy as np

class LogisticRegression(Module):
    def __init__(self, in_features, out_features):
        self.linear = Linear(in_features, 2)
        self.linear2 = Linear(2, 1)
        self.sigmoid = Sigmoid()
        self.elu = ELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.elu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

# model = LogisticRegression(3, 2)
input = Tensor(np.array([[1], [0], [0]]))
# output = model(input)

loss_fn = MSELoss()
# loss = loss_fn(Tensor(np.ones((2, 1))), output)
# model.zero_grad()
# loss.backward()
# print("input 1\n", model.linear.weight.grad)
# g1 = model.linear.weight.grad

# in_2 = Tensor(0 * np.ones((3, 1)))
# out = model(in_2)
# l = loss_fn(Tensor(np.ones((2, 1))), out)
# l.backward()
# print("input 1 + input 2\n", model.linear.weight.grad)
# g12 = model.linear.weight.grad

# out = model(in_2)
# model.zero_grad()
# l = loss_fn(Tensor(np.ones((2, 1))), out)
# l.backward()
# print("input 2\n", model.linear.weight.grad)
# g2 = model.linear.weight.grad

print("TESTING LOOPY MODELS")

class LoopModel(Module):
    def __init__(self, in_features, hidden_features, out_features, N=2):
        self.linear_head = Linear(in_features, hidden_features)
        self.elu_head = ELU()
        self.linear_stack = [Linear(hidden_features, hidden_features) for _ in range(N)]
        # self.linear_stack = []
        # for _ in range(N):
        #     self.linear_stack.append(Linear(hidden_features, hidden_features))
        # self.elu_stack = [ELU() for _ in range(N)]
        self.elu = ELU()
        # self.elu_stack = []
        # for _ in range(N):
        #     self.elu_stack.append(ELU())
        self.linear_tail = Linear(hidden_features, out_features)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        x = self.linear_head(x)
        x = self.elu_head(x)

        for i in range(len(self.linear_stack)):
            x = self.linear_stack[i](x)
            # x = self.elu_stack[i](x)
            x = self.elu(x)
        
        x = self.linear_tail(x)
        x = self.sigmoid(x)
        return x

loopy_model = LoopModel(3, 2, 1)
out = loopy_model(input)
target = Tensor([1])
l = loss_fn(target, out)
l.backward()

optimizer = SGD_Momentum(loopy_model.parameters(), lr=0.01)

for _ in range(1000):
    out = loopy_model(input)
    l = loss_fn(target, out)
    print(l.data)
    l.backward()
    optimizer.step()