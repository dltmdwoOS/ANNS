from layer import *
from functions import numerical_grad
from collections import OrderedDict
from activation_function import *
from optimizer import *

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            'W1':2 * np.random.randn(input_size, hidden_size) / np.sqrt(input_size),
            'b1':np.zeros(hidden_size),
            'W2':2 * np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size),
            'b2':np.zeros(output_size)
        }
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine_layer(self.params['W1'], self.params['b1'])
        self.layers["ReLU1"] = ReLU_layer()
        self.layers["Affine2"] = Affine_layer(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def infer(self, x):
        y_hat = self.predict(x)
        return np.argmax(softmax(y_hat))
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        for key in list(self.params.keys()):
            grads = {}
            grads[key] = numerical_grad(loss_W, self.params[key])
        return grads
    
    def gradient(self, x, t):
        #순전파
        self.loss(x, t)
        
        #역전파
        dout = 1
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {
            'W1':self.layers['Affine1'].dW,
            'b1':self.layers['Affine1'].db,
            'W2':self.layers['Affine2'].dW,
            'b2':self.layers['Affine2'].db
        }

        return grads
    
    
class MultiLayerNetwork:
    def __init__(self, optimizer="sgd", info={"Affine1":0, "SoftmaxWithLoss":0}): # OneLayerNetwork
        self.num_layers = 0
        self.num_hidden_layers = -1
        self.params = {}
        self.layers = OrderedDict()
        self.lastLayer = None
        for key in list(info.keys()):
            if "Affine" in key:
                self.num_layers += 1
                self.num_hidden_layers += 1
                shape = info[key]
                self.params[f"W{self.num_layers}"] = 2 * np.random.randn(*shape) / np.sqrt(shape[0])
                self.params[f"b{self.num_layers}"] = np.zeros(shape[1])
                self.layers[key] = Affine_layer(self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"])
                
            elif "DropOut" in key:
                self.layers[f"DropOut{self.num_layers}"] = DropOut()
            
            elif "ReLU" in key:
                self.layers[f"ReLU{self.num_layers}"] = ReLU_layer()
                
            elif "Sigmoid" in key:
                self.layers[f"ReLU{self.num_layers}"] = Sigmoid_layer()

            elif "Softmax" in key:
                self.lastLayer = SoftmaxWithLoss()
                break
        
        self.optimizer = optimizer
        self.loss_log = []
        self.train_acc_log = []
        self.test_acc_log = []
        
    def predict(self, x):
        '''
        Input : x(sample)
        Output : traversed x for all layers but last layer(softmax).
        '''
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def infer(self, x):
        '''
        Input : x(sample)
        Output : predicted class
        '''
        y_hat = self.predict(x)
        return np.argmax(y_hat)
    
    def loss(self, x, t):
        '''
        Input : X(data), t(label)
        Output : Average loss of current batch with current params
        '''
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        for key in list(self.params.keys()):
            grads = {}
            grads[key] = numerical_grad(loss_W, self.params[key])
        return grads
    
    def gradient(self, x, t):
        #순전파
        self.loss(x, t)
        
        #역전파
        dout = 1
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for i in range(1, self.num_layers+1):
            grads[f"W{i}"] = self.layers[f"Affine{i}"].dW
            grads[f"b{i}"] = self.layers[f"Affine{i}"].db

        return grads
    
    def fit(self, X_train, y_train, batch_size, epoch=1000, learning_rate=0.01, moment_const=0.8):
        self.loss_log, self.train_acc_log = optimizer_list[self.optimizer](self, X_train, y_train, batch_size, epoch, learning_rate, moment_const)
        return f"final loss : {self.loss_log[-1]} / final train accuracy : {self.train_acc_log[-1]}"