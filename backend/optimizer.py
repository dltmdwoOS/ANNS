import numpy as np
from functions import *

def gradient_descent(network, X_train, y_train, batch_size, epoch, learning_rate=0.01, moment_const=0.8):
    train_size = X_train.shape[0]
    for i in range(epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = X_train[batch_mask]
        y_batch = y_train[batch_mask]
        
        grad = network.gradient(X_batch, y_batch)
        for key in list(network.params.keys()): # gradient descent
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(X_batch, y_batch)
        network.loss_log.append(loss)
        
        train_acc = network.accuracy(X_train, y_train)
        network.train_acc_log.append(train_acc)
        
    return network.loss_log, network.train_acc_log
        
def momentum(network, X_train, y_train, batch_size, epoch, learning_rate=0.01, moment_const=0.8):
    train_size = X_train.shape[0]
    moment = {}
    for key in list(network.params.keys()):
        moment[key] = np.zeros_like(network.params[key])
        
    for i in range(epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = X_train[batch_mask]
        y_batch = y_train[batch_mask]
        
        grad = network.gradient(X_batch, y_batch)
        for key in list(network.params.keys()): # gradient descent
         moment[key] = moment_const*moment[key] - learning_rate*grad[key]
         network.params[key] += moment[key]
        
        loss = network.loss(X_batch, y_batch)
        network.loss_log.append(loss)
        train_acc = network.accuracy(X_train, y_train)
        network.train_acc_log.append(train_acc)
        
    return network.loss_log, network.train_acc_log

optimizer_list = {
    "sgd":gradient_descent,
    "momentum":momentum
}