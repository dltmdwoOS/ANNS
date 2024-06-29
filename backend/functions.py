import numpy as np

# Loss functions
def mean_square_error(y, t):
    batch_size = y.shape[0]
    return np.sum((y-t)**2) / (2*batch_size)

def cross_entropy_error(y, t):
    if y.ndim == 1: # (1,) -> (1, size)
        y = np.reshape(1, y.size)
        t = np.reshape(1, t.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fx1 = f(x)  # f(x + h)
        x[idx] = tmp_val - h
        fx2 = f(x)  # f(x - h)
        grad[idx] = (fx1 - fx2) / (2 * h)
        x[idx] = tmp_val 
        
        it.iternext()

    return grad
