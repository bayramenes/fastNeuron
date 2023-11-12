# this is a file containing the activation functions we will be using 



from typing import Any
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
    

class linear:
    def __repr__(self) -> str:
        return "Linear"
    def __call__(self,z):
        return z
    
    def derivative(self):
        return 1
    
    

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
        # # calculate the jacobian for each training example
        # J = -self.output[...,None] * self.output[:,None,:]
        # iy,ix = np.diag_indices_from(J[0])
        # J[:,iy,ix] = self.output * (1 - self.output)
        # return J.sum(axis=1)
        return 1
    


    