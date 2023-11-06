
# this is a file that will contain implementation of different cost functions that are used in the wild

import numpy as np

class BinaryCrossEntropy:

    def __repr__(self) -> str:
        return "Binary Cross Entropy"
    
    

    def __call__(self,output:np.ndarray,labels:np.ndarray) -> np.float64:
        return -1 / output.shape[0] * np.sum(labels * np.log(output + 1e-15) + (1 - labels) * np.log(1 - output + 1e-15))
    
    
    
    def initial_derivative(self,output:np.ndarray,labels:np.ndarray) -> np.ndarray:
        return output - labels



class CategoricalCrossEntropy:
    def __repr__(self) -> str:
        return "Categorical Cross Entropy"
    
    def __call__(self,output:np.ndarray,labels:np.ndarray) -> np.float64:
        pass

    
    def initial_derivative(self):
        pass



class MSE:
    def __repr__(self) -> str:
        return "Mean Squared Error"
    
    def __call__(self,output:np.ndarray,labels:np.ndarray) -> np.float64:
        return (1/2) * np.mean(np.square(output - labels))

    
    def initial_derivative(self,output:np.ndarray,labels:np.ndarray) -> np.ndarray:
        return output - labels
        