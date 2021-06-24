import numpy as np
import numba as nb

@nb.jit(nopython=True)
def func(x):
    y = x#x.transpose()  # or x.T
    return y

x = np.random.normal(size=(4,4))
x_t = func(x)
