from layers import Identity
from tensor import Tensor

import numpy as np

identity = Identity()

x = Tensor(np.zeros((2, 1)))
y = identity(x)
y.backward()

print(identity.name)