
# this is a file that will contain implementation of different cost functions that are used in the wild

import numpy as np

class BinaryCrossEntropy:
    """
    implemenationn of binary cross entropy
    this loss function shall be used with sigmoid activation function in the last layer with one output node

    :param output: output of the last layer of the model
    :param labels: labels of the training examples
    :return: the loss value
    """

    def __repr__(self) -> str:
        return "Binary Cross Entropy"
    
    

    def __call__(self,output:np.ndarray,labels:np.ndarray) -> np.float64:
        return -1 / output.shape[0] * np.sum(labels * np.log(output + 1e-15) + (1 - labels) * np.log(1 - output + 1e-15))
    
    
    
    def derivative(self,output:np.ndarray,labels:np.ndarray) -> np.ndarray:
        """
        initial derivative to start backprop this will be the multiplicaition  of the derivative of the loss function and activation function of the last layer
        :return float representing the initial derivative
        
        for a more involved and math heavy derivation check out this
        https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
        """
        return (output - labels) / (output * (1 - output) + 1e-100)



class CategoricalCrossEntropy:
    """
    implemenation of categorical cross entropy 
    this loss function shall be used with softmax activation function in the last layer that's why
    it is also called softmax loss
    the difference between it and binary cross entropy is that binary cross entropy is used with sigmoid activation function
    and one node output where as categorical cross entropy works with multiple output nodes

    :param output: output of the last layer of the model
    :param labels: labels of the training examples
    :return: the loss value
    """
    def __repr__(self) -> str:
        return "Categorical Cross Entropy"
    
    def __call__(self,output:np.ndarray,labels:np.ndarray) -> np.float64:
        return np.sum(-labels * np.log(output + 1e-15))

    
    def derivative(self,output:np.ndarray , labels:np.ndarray) -> np.ndarray:
        """
        initial derivative to start backprop this will be the multiplicaition  of the derivative of the loss function and activation function of the last layer
        :return float representing the initial derivative
        
        for a more involved and math heavy derivation check out this
        https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        """

        # this is the derivative of categorical cross entropy with respect to the raw outputs
        # i have done it this way because it is easier
        return output - labels




class MSE:
    def __repr__(self) -> str:
        return "Mean Squared Error"
    
    def __call__(self,output:np.ndarray,labels:np.ndarray) -> np.float64:
        return (1/2) * np.mean(np.square(output - labels))

    
    def derivative(self,output:np.ndarray,labels:np.ndarray) -> np.ndarray:
        """
        initial derivative to start backprop this will be the multiplicaition  of the derivative of the loss function and activation function of the last layer
        :return float representing the initial derivative
        
        for a more involved and math heavy derivation check out this
        https://math.stackexchange.com/questions/3713832/derivative-of-mean-squared-error
        """
        return output - labels
    


        