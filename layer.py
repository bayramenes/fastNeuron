# this file will be used to create the layer class

# a layer in a neural network is a data structure that holds a bunch of perceptrons(neurons)
# it is a higher level abstraction to abstract some of the code we have to write to propogate forward and backward through the network
# in other words it is a container for the perceptrons

# import dependencies
from perceptron import perceptron
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
        assert activation in ["sigmoid","tanh","relu","none"],"activation function must be one of the following: sigmoid,tanh,relu,step"


        # save values to the object
        self.input_size = input_size
        self.units = units
        self.activation = activation
        self.outputs = None




        # initialize as many perceptrons as given
        self.perceptrons = [perceptron(input_size, activation) for _ in range(units)]

        # initialize the weight of the perceptrons according to Xavier/Glorot Initialization
        sd = np.sqrt(6/(input_size + units))


        # a matrix that has all of the values of the weights of all of the units
        self.weights = np.random.normal(0,sd,size=(input_size,units))
        self.biases = np.random.normal(0,sd,size=(1,units))

        # # by default i will use the normal distribution
        # for perceptron_unit in self.perceptrons:
        #     perceptron_unit.weights = np.random.normal(0,sd,size=(input_size,1))
        #     perceptron_unit.bias = np.random.normal(0,sd,size=1)


    def __repr__(self) -> str:
        return f"layer with {self.units} units, {self.input_size} inputs and {self.activation} activation function"
    
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    # forward propagation
    def forward(self,X:np.ndarray) -> np.ndarray:
        return layer.sigmoid(X @ self.weights + self.biases)

    
    def backward(self,pervious_derivatives:float):
        """
        this function returns three arrays

        1. an array of the gradients of the weights
        2. an array of the gradients of the biases
        3. an array of the graidents of the inputs

        """
        # we will store weight gradients in a 2d array where each column represents the grads of that particulat unit

        weight_grads = np.array([
            perceptron_unit.weight_grads(pervious_derivatives) for perceptron_unit in self.perceptrons
        ])

        # we will store bias gradients in a row vector where each column entry is gradient of the corresponding unit bias

        bias_grads = np.array(
            [
                perceptron_unit.bias_grad(pervious_derivatives) for perceptron_unit in self.perceptrons
            ]
        )


        # we will store input gradient in a 2d array where each column represents the gradient of the corresponding input

        input_grads = np.array(
            [
                perceptron_unit.input_grads(pervious_derivatives) for perceptron_unit in self.perceptrons
            ]
        )


        return weight_grads , bias_grads , input_grads
        
    
    def raw_outputs(self) -> np.ndarray:
        return np.array([[perceptron_unit.raw_output for perceptron_unit in self.perceptrons]])
        


    


