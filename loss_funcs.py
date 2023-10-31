
# this is a file that will contain implementation of different cost functions that are used in the wild

import numpy as np

class BinaryCrossEntropy:

    @staticmethod 
    def calculate(output:np.ndarray,labels:np.ndarray) -> np.float64:
        return -1 / output.shape[0] * np.sum(labels * np.log(output) + (1 - labels) * np.log(1 - output))
    
    
    @staticmethod
    def initial_derivative(output:np.ndarray,labels:np.ndarray) -> np.ndarray:
        return output - labels



class CategoricalCrossEntropy:
    @staticmethod 
    def calculate(output:np.ndarray,labels:np.ndarray) -> np.float64:
        pass

    @staticmethod
    def initial_derivative():
        pass



class MSE:
    @staticmethod 
    def calculate(output:np.ndarray,labels:np.ndarray) -> np.float64:
        return (1/2) * np.mean(np.square(output - labels))

    @staticmethod
    def initial_derivative(output:np.ndarray,labels:np.ndarray) -> np.ndarray:
        return output - labels
        