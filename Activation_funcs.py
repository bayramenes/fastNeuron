# this is a file containing the activation functions we will be using 

import numpy as np


class sigmoid:
    def __init__(self) -> None:
        self.output = None

    def __repr__(self) -> str:
        return "Sigmoid"

    def __call__(self,z):
        e = np.exp(z)
        self.output = e / (1 + e)
        return self.output
    
    def derivative(self):
        return self.output * (1 - self.output + 1e-100)
    
    def predict(self):
        return (self.output >= 0.5).astype(int)
    


class tanh:
    def __init__(self) -> None:
        self.output = None

    def __repr__(self) -> str:
        return "Tanh"


    def __call__(self,z):
        e2 = np.exp(z) ** 2
        self.output = (e2 - 1) / (e2 + 1)
        return self.output
    def derivative(self):
        return 1 - (self.output ** 2)
    

    def predict(self):
        return (self.output >= 0).astype(int)
    


class relu:
    def __init__(self,leak=0) -> None:
        self.output = None 
        self.leak = leak

    def __repr__(self) -> str:
        return "ReLU"

    def __call__(self,z):
        self.output = np.maximum(z*self.leak,z)
        return self.output
    
    def derivative(self):
        return np.where(self.output > 0, 1, self.leak)
    
    def predict(self):
        return self.output
    

class linear:

    def __init__(self) -> None:
        self.output = None
    def __repr__(self) -> str:
        return "Linear"
    def __call__(self,z):
        self.output = z
        return self.output
    
    def derivative(self):
        return 1
    
    def predict(self):
        return self.output

    

class softmax:
    def __init__(self) -> None:
        self.output = None

    def __repr__(self) -> str:
        return "Softmax"


    def __call__(self,z):
        e = np.exp(z - np.max(z,axis=1,keepdims=True))
        self.output = e / np.sum(e,axis=1,keepdims=True)
        return self.output
    
    def derivative(self):
        # since softmax is only used with categorical cross entropy loss and because it is hard to 
        # get an easy derivative for softmax alone and instead it is pretty simple to get a derivative for categorical cross entropy with 
        # respect to the raw outputs immediatly because of the simplification that occurs i will just return 1 and let
        # the derivative of categorical cross entropy handle the rest
        return 1
    
    def predict(self):
        z = np.zeros(self.output.shape)
        # get the index of the maximum value in each row and assign it the value 1 others will be 0
        z[np.arange(self.output.shape[0]),np.argmax(self.output , axis=1)] = 1
        return z
        

    
    