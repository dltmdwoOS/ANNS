import numpy as np

def step_function(x):
    if x>0:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return x if x > 0 else 0

def softmax(x): 
    if x.ndim == 2: # When x is 2D matrix [[#, #, ... , #], [#, #, ... , #], ... , [#, #, ... , #]]
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # When x is 1D vector [#, #, ..., #]
    return np.exp(x) / np.sum(np.exp(x))
