# this is a class that initiates a neural network and gives high level control over
# it
from loss_funcs import *
from Optimizers import *
from layer import layer
import numpy as np


class neural_network:
    def __init__(self) -> None:
        # initialize values
        self.layers = []
        self.input_size = None
        self.outputs = []

        # default optimizer and loss functions
        self.optimizer = BatchGradientDescent()
        self.cost = MSE()

    # accept a list of layers and set as the layers of the neural network
    def sequential(self, layers: list[layer]):
        self.layers = layers
        self.input_size = layers[0].input_size
        return self

    def forward(self, inputs: np.ndarray):
        # iterate through the layers
        for layer_ in self.layers:
            self.outputs.append(inputs)
            inputs = layer_.forward(inputs)
        # we have to add the outputs of the last layer manually
        self.outputs.append(inputs)
        return inputs

    def get_weights(self):
        return [layer_.weights for layer_ in self.layers]

   
    def compile(self, optimizer: object, Cost: object):  # cost function object
        self.optimizer = optimizer
        self.cost = Cost
        return self

    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float, epochs: int):
        model, costs = self.optimizer(
            self,
            self.cost,
            X,
            Y,
            learning_rate,
            epochs,
        )
        return model, costs
    
    def summary(self):
        """
        print a summary of the model including weight biases and activations
        """
        print("Summary of the neural network")
        for layer_ in self.layers:
            print(layer_)
            print("weights:")
            print(layer_.weights)
            print("biases:")
            print(layer_.biases)
            print()
            print('-'*50)
            print()
        print(f"input size : {self.input_size}")
        print(f"output size : {self.layers[-1].units}")
        print(f"optimizer : {self.optimizer}")
        print(f"cost function : {self.cost}")
        return self

    def predict(self, X: np.ndarray):
        """
        given an input dataset provide the output of the model
        """
        return (self.forward(X) >= 0.5).astype(int)
