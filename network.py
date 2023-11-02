# this is a class that initiates a neural network and gives high level control over
# it
from loss_funcs import *
from Optimizers import Optimizers
from layer import layer
import numpy as np


class neural_network:
    def __init__(self) -> None:
        # initialize values
        self.layers = []
        self.input_size = None
        self.outputs = []

        # default optimizer and loss functions
        self.optimizer = Optimizers.BatchGradientDescent
        self.cost = MSE()

    # accept a list of layers and set as the layers of the neural network
    def sequential(self, layers: list[layer]):
        self.layers = layers
        self.input_size = layers[0].input_size
        return self

    def forward(self, inputs: np.ndarray):
        # iterate through the layers

        for layer in self.layers:
            self.outputs.append(inputs)
            inputs = layer.forward(inputs)

        # we have to add the outputs of the last layer manually
        self.outputs.append(inputs)
        return inputs

    def get_weights(self):
        return [layer_.weights for layer_ in self.layers]

    # def backward(self, initial_derivatives: np.ndarray):
    #     # for now i will assume it is only one unit so i will apply binary cross-entropy
    #     # initial_derivatives is a matrix that contains the error terms of all of the training samples which are the starting initial_derivatives

    #     # NOTE: this algorithm can be implemented using recursion or using iterations
    #     #  i chose to use iterations for performance purposes...

    #     # since this is called backpropagation we have to go backwards through the layers

    #     wgrads = []
    #     bgrads = []
    #     for index in range(len(self.layers) - 1, -1, -1):
    #         layer_wgrads = (
    #             1
    #             / initial_derivatives.shape[0]
    #             * np.matmul(self.outputs[index].T, initial_derivatives)
    #         )
    #         layer_bgrads = (
    #             1 / initial_derivatives.shape[0] * np.sum(initial_derivatives, axis=0)
    #         )
            
    #         initial_derivatives = np.matmul(
    #             initial_derivatives, self.layers[index].weights.T
    #         )
    #         if self.layers[index].activation == "sigmoid":
    #             initial_derivatives *= self.outputs[index] * (1 - self.outputs[index])
    #         elif self.layers[index].activation == "relu":
    #             initial_derivatives *= (self.outputs[index] > 0).astype(int)
    #         elif self.layers[index].activation == "tanh":
    #             initial_derivatives *= 1 - self.outputs[index] * self.outputs[index]

    #         wgrads = [layer_wgrads] + wgrads
    #         bgrads = [layer_bgrads] + bgrads

    #     return wgrads, bgrads

    def compile(self, optimizer: Optimizers, Cost: object):  # cost function object
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
        print("Summary of the neural network")
        for layer in self.layers:
            print(layer)
            print("weights:")
            print(layer.weights)
            print("biases:")
            print(layer.biases)
            print()
            print('-'*50)
            print()
        print(f"input size : {self.input_size}")
        print(f"output size : {self.layers[-1].units}")
        print(f"optimizer : {self.optimizer}")
        print(f"cost function : {self.cost}")
        return self

    def predict(self, X: np.ndarray):
        return (self.forward(X) >= 0.5).astype(int)
