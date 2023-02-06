from types import DynamicClassAttribute
import numpy as np
import numba as nb
from abc import abstractmethod
from .tensor import Tensor

class Module:
    def __init__(self):
        self.train_status = True
        self.grad_out = None

    @abstractmethod
    def get_grad_fn(self, *inputs):
        def grad(psi=None) -> Tensor:
            # out = self.forward(*inputs)
            return self.grad_out.backward(psi=psi)
        return grad

    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        pass
    
    def __call__(self, *inputs):
        self.grad_out = self.forward(*inputs)
        grad = self.get_grad_fn(*inputs)
        return Tensor(self.grad_out.data, gradFn=grad)
    
    def zero_grad(self):
        vars = [v for v in self.__dict__.values() if isinstance(v, Module) or isinstance(v, Tensor)]
        for var in vars:
            var.zero_grad()

    def train(self):
        self.train_status = True
        vars = [v for v in self.__dict__.values() if isinstance(v, Module)]
        for var in vars:
            var.train()
        
    def eval(self):
        self.train_status = False
        vars = [v for v in self.__dict__.values() if isinstance(v, Module)]
        for var in vars:
            var.eval()

    def parameters(self):
        # this is very good code, not at all messy
        vars = []
        for n, var in self.__dict__.items():
            # take modules/tensors and add them to the parameters
            if isinstance(var, Module) or isinstance(var, Tensor):
                vars += [(n, var)]
            # unwrap lists of modules/tensors and add them to the list of usable variables
            elif isinstance(var, list):
                vars += [(f"{n}_{i}", v) for i, v in enumerate(var) if isinstance(v, Tensor) or isinstance(v, Module)]
        return {n: v.parameters() if isinstance(v, Module) else v for n, v in vars}
    
    def state_dict(self):
        vars = []
        for n, var in self.__dict__.items():
            # take modules/tensors and add them to the parameters
            if isinstance(var, Module) or isinstance(var, Tensor):
                vars += [(n, var)]
            # unwrap lists of modules/tensors and add them to the list of usable variables
            elif isinstance(var, list):
                vars += [(f"{n}_{i}", v) for i, v in enumerate(var) if isinstance(v, Tensor) or isinstance(v, Module)]
        return {n: v.state_dict() if isinstance(v, Module) else v.data.tolist() for n, v in vars}
    
    def load_state_dict(self, state_dict):
        params = self.parameters()

        def update_state(params, state_dict):
            for k, v in params.items():
                if isinstance(v, dict) and k in state_dict:
                    update_state(v, state_dict[k])
                elif isinstance(v, Tensor) and k != 'out':
                    # this is prob a massive abuse, but hey it works so im not gonna question it
                    params[k].data = np.array(state_dict[k])
                elif k != 'out':
                    raise RuntimeWarning("Found Invalid State Dictionary")
        
        update_state(params, state_dict)

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
    def __init__(self, in_features, out_features, initialization="kaiming", drop_connect=False, p=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.drop_connect = drop_connect
        # keep probability
        self.p = p

        self.w_mask = None
        self.b_mask = None

        if initialization == "xavier":
            self.weight = Tensor(np.random.uniform(-1, 1, size=(out_features, in_features)) / np.sqrt(in_features))
            self.bias = Tensor(np.random.uniform(-1, 1, size=(out_features, 1)) / np.sqrt(in_features))
        elif initialization == "kaiming":
            self.weight = Tensor(np.random.randn(out_features, in_features) * np.sqrt(2 / in_features))
            self.bias = Tensor(np.random.randn(out_features, 1) * np.sqrt(2 / in_features))
        elif initialization == "lecun":
            std = np.sqrt(1/in_features)
            self.weight = Tensor(np.random.normal(0, std, (out_features, in_features)))
            self.bias = Tensor(np.random.normal(0, std, (out_features, 1)))
        elif initialization == "zeros":
            self.weight = Tensor(np.zeros((out_features, in_features)))
            self.bias = Tensor(np.zeros((out_features, 1)))
        else:
            raise ValueError("Invalid initialization")

    def get_grad_fn(self, x_in: Tensor):
        def linear_grad(psi=Tensor(np.ones_like(x_in))) -> Tensor:
            x = x_in.data
            p = psi.data

            if self.out_features == 1:
                dW = (p * x).T
            else:
                dW = p @ x.T
            dB = p

            if self.w_mask is not None:
                dW *= self.w_mask
            if self.b_mask is not None:
                dB *= self.b_mask

            if self.train_status:
                self.weight.update_grad(dW)
                self.bias.update_grad(dB)

            return Tensor(self.weight.data.T @ psi.data, gradFn=x_in.gradFn)
        return linear_grad

    # TODO: Add safety checks for batching
    def forward(self, x: Tensor) -> Tensor:
        W = np.copy(self.weight.data)
        b = np.copy(self.bias.data)

        if self.drop_connect and self.train_status:
            self.w_mask = np.random.binomial(1, self.p, self.in_features * self.out_features).reshape(W.shape)
            self.b_mask = np.random.binomial(1, self.p, self.out_features).reshape(b.shape)

            # print("prior: ", W @ x.data + b)

            W *= self.w_mask
            b *= self.b_mask
            return Tensor(W @ x.data + b)
        elif self.drop_connect:
            mean = self.p * (W @ x.data + b)
            std = self.p * (1 - self.p) * ((W * W) @ (x.data * x.data) + (b * b))
            sample = np.random.normal(mean, std)
            return Tensor(sample)

        return Tensor(W @ x.data + b)
    
class ELU(Module):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def get_grad_fn(self, x_in: Tensor):
        x = x_in.data
        def elu_grad(psi=Tensor(np.ones_like(x_in))) -> Tensor:
            g = np.where(x > 0, 1, self.alpha * np.exp(x))

            return Tensor(g.reshape((-1, 1)) * psi.data, gradFn=x_in.gradFn)
        return elu_grad

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1)))

class SELU(Module):
    def __init__(self, alpha=1.6732632423543772848170429916717,
                        scale=1.0507009873554804934193349852946):
        self.alpha = alpha
        self.scale = scale

    def get_grad_fn(self, x_in: Tensor):
        x = x_in.data
        def selu_grad(psi=Tensor(np.ones_like(x_in.data))) -> Tensor:
            g = np.where(x > 0, self.scale, self.scale * self.alpha * np.exp(x))
            # g = np.diag(g)

            return Tensor(g.reshape((-1, 1)) * psi.data, gradFn=x_in.gradFn)
        return selu_grad
    
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.where(x.data > 0, self.scale * x.data, self.scale * self.alpha * (np.exp(x.data) - 1)))

class GeLU(Module):
    def __init__(self):
        pass

    def get_grad_fn(self, x_in: Tensor):
        x = x_in.data
        def gelu_grad(psi=Tensor(np.ones_like(x_in.data))) -> Tensor:
            x3 = x ** 3
            u = (0.0356774 * x3) + (0.797885 * x)
            sech2 = (1 / np.cosh(u)) ** 2
            g = (0.5 * np.tanh(u)) + (((0.0535161 * x3) + (0.398942 * x)) * sech2) + 0.5

            return Tensor(g.reshape((-1, 1)) * psi.data, gradFn=x_in.gradFn)
        return gelu_grad
    
    def forward(self, x: Tensor) -> Tensor:
        npx = x.data
        y = 0.5 * npx * (1 + np.tanh(np.sqrt(2 / np.pi) * (npx + (0.044715 * (npx ** 3)))))
        return Tensor(y)

class Sigmoid(Module):
    def __init__(self):
        self.out = None

    def get_grad_fn(self, x_in: Tensor):
        def sigmoid_grad(psi=Tensor(np.ones_like(x_in.data))) -> Tensor:
            if self.out is not None:
                # ravel is pain
                g = (self.out.data * (1 - self.out.data) * psi.data)
                # print(self.out.data)
                # print("Sigmoid Grad: ", g)
                # if g.shape[-1] == 1:
                #     g = (g.ravel())
                # out = Tensor([np.diag(g[:, i]) for i in range(g.shape[-1])], gradFn=x_in.gradFn)
                out = Tensor(g, gradFn=x_in.gradFn)
                return out
            out = np.exp(-x_in.data) / np.square(1 + np.exp(-x_in.data))
            return Tensor(out @ psi.data, gradFn=x_in.gradFn)
        return sigmoid_grad
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.data
        out = np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
        self.out = Tensor(out)
        return self.out

class Dropout(Module):
    def __init__(self, p=0.2):
        self.p = p
        self.mask = None
    
    def get_grad_fn(self, x_in: Tensor):
        x = x_in.data
        mask = self.mask
        def dropout_grad(self, psi=Tensor(np.ones_like(x))) -> Tensor:
            out = psi.data
            if mask is not None:
                out = mask * out
            return Tensor(out, gradFn=x_in.gradFn)
        return dropout_grad
    
    def forward(self, x):
        i = x.data
        if self.train_status:
            self.mask = np.random.choice([0, 1], size=i.shape, p=[self.p, 1-self.p])
            i = self.mask * i
        return Tensor(i)