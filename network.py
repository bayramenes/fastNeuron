# this is a class that initiates a neural network and gives high level control over
# it

from layer import layer
import numpy as np
class neural_network:
    def __init__(self) -> None:

        # initialize values
        self.layers = []
        self.input_size = None
        self.outputs = []

    # accept a list of layers and set as the layers of the neural network
    def sequential(self,layers:list[layer]):
        self.layers = layers
        self.input_size = layers[0].input_size
        return self
    


    def forward(self,inputs:np.ndarray):
        # iterate through the layers

        
        for layer in self.layers:
            self.outputs.append(inputs)
            inputs = layer.forward(inputs)
        return inputs
    
    def get_weights(self):
        return [layer_.weights for layer_ in self.layers]
    
    def backward(self,initial_derivatives:np.ndarray):

        # for now i will assume it is only one unit so i will apply binary cross-entropy
        # initial_derivatives is a matrix that contains the error terms of all of the training samples which are the starting initial_derivatives


        # NOTE: this algorithm can be implemented using recursion or using iterations
        #  i chose to use iterations for performance purposes...



        # since this is called backpropagation we have to go backwards through the layers

        wgrads = []
        bgrads = []
        for index in range(len(self.layers) - 1 , -1 ,-1):
            layer_wgrads = np.matmul(self.outputs[index].T,initial_derivatives)
            layer_bgrads = np.sum(initial_derivatives , axis= 0)
            # layer_igrads = initial_derivatives * self.layers[index].weights
            # for now i am assuming that we are only going to use the sigmoid function as an activation function
            print(f"self.outputs[index].shape:{self.outputs[index].shape}")
            initial_derivatives = np.matmul(initial_derivatives,self.layers[index].weights.T) * (self.outputs[index] * (1 - self.outputs[index]))
            wgrads = [layer_wgrads] + wgrads
            bgrads = [layer_bgrads] + bgrads
            print(f"propogated through layer{index}")



        return wgrads , bgrads






    


        
        