# this file will be used to create the layer class

# a layer in a neural network is a data structure that holds a bunch of perceptrons(neurons)
# it is a higher level abstraction to abstract some of the code we have to write to propogate forward and backward through the network
# in other words it is a container for the perceptrons

# import dependencies
import numpy as np


class layer:

    # initialize the layer
    def __init__(
                self,
                input_size:int,
                units:int,
                activation:str,
            ) -> None:
        

        # make sure values are correct
        assert isinstance(input_size,int),"input_number must be an integer represeting the number of input_number"
        assert isinstance(activation,str),"activation function must be a string"
        assert activation in ["sigmoid","tanh","relu","none","leaky-relu"],"activation function must be one of the following: sigmoid,tanh,relu,step"


        # save values to the object
        self.input_size = input_size
        self.units = units
        self.activation = activation
        self.outputs = None

        # initialize the weight of the perceptrons according to Xavier/Glorot Initialization
        sd = np.sqrt(1/(input_size + units))
        if activation == "sigmoid": sd *= np.sqrt(6)
        elif activation == "relu" or activation == "leaky-relu": sd *= np.sqrt(2)


        # a matrix that has all of the values of the weights of all of the units
        self.weights = np.random.normal(0,sd,size=(input_size,units))
        self.biases = np.random.normal(0,sd,size=(1,units))

    def __repr__(self) -> str:
        return f"layer with {self.units} units, {self.input_size} inputs and {self.activation} activation function"
    
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    @staticmethod
    def relu(z):
        return np.maximum(0,z)
    @staticmethod
    def leaky_relu(z):
        return np.maximum(0.1*z,z)

    # forward propagation
    def forward(self,X:np.ndarray) -> np.ndarray:
        raw = X @ self.weights + self.biases
        # print(raw)
        if self.activation == "sigmoid": return layer.sigmoid(raw)
        elif self.activation == "tanh": return layer.tanh(raw)
        elif self.activation == "relu": return layer.relu(raw)
        elif self.activation == "leaky-relu": return layer.leaky_relu(raw)


    


