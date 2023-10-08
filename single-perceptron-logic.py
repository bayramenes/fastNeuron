# this is a test file that will be used to create a small perceptron which somewhat is also called a neuron
# it is basically the notion of having multiple inputs each with there weight alongside with a bias that then gets fed into 
# a so called "activation function" and in return the perceptron outputs a value

# we can think of the inputs as a vector that contains the values of the inputs
# e.g. [input1 , input2 , input3]
# the weights are also a vector that contains the weights of the inputs
# e.g. [weight1 , weight2 , weight3]
# the bias is the value that gets added to the weighted sum of the inputs
# now the process goes this way:
# multiply each input with its corresponding weight and then sum all the values and lastly add the bias...
# in more mathy lingo take the DOT product of the input and weight vectors and add a bais.
# i will be implementing a couple of different activation functions mainly (sigmoid,tanh,relu,)

from numpy.random import randint
from numpy import ndarray,exp
import numpy as np


# create a perceptron class

class perceptron:
    def __init__(
                    self,inputs:int,
                    activation_function:str
                 ) -> None:
        

        # make sure that the values given are correct
        assert isinstance(inputs,int),"inputs must be an integer represeting the number of inputs"
        assert isinstance(activation_function,str),"activation function must be a string"
        assert activation_function in ["sigmoid","tanh","relu","step"],"activation function must be one of the following: sigmoid,tanh,relu,step"


        self.inputs = inputs
        self.activation_function = activation_function
        self.weights = None
        self.bias = None


    # set the weights either randomly or by the given values in the following way
    # weights should be a 2 dimensional array with each row being the input and each column being the weight index
    # so the input for weight1 of input2 is weights[1][0]
    # and the input for weight2 of input2 is weights[1][1]
    # and so on

    def set_weights(self,weights:list[float] = None) -> None:
        # not weights were provided so we will generate them randomly
        if weights is None:
            self.weights = randint(-10,10,size=(self.inputs,))
            return
        # if a set of weight is given we have to make sure that it is the correct shape and type
        # type should be an ndarray 
        # size should be (inputs,1)
        # since we have inputs number of rows and 1 weight for each input

        assert isinstance(weights,ndarray),"weights must be a ndarray"
        assert weights.shape == (self.inputs,),"weights must be a ndarray of shape (inputs,)"

        self.weights = weights



    # get the current weight of the perceptron
    def get_weights(self) -> ndarray:
        return self.weights
    

    def set_bias(self,bias:float = None) -> None:

        if bias is None : self.bias = randint(-10,10) ; return
        # make sure that the bias is a float
        assert isinstance(bias,float),"bias must be a float"
        self.bias = bias


    def get_bias(self):
        return self.bias
    
    def calculate(self,inputs:list[float]) -> float :
        # make sure that input is the correct size and type
        assert isinstance(inputs,list),"inputs must be a list"
        assert len(inputs) == self.inputs,"inputs must be of the same size as the number of inputs"
        # make sure that weight and biases are set
        assert self.weights is not None,"weights must be set"
        assert self.bias is not None,"bias must be set"

        # we now calculate the value of the perceptron
        raw_value = sum(w*i for w,i in zip(self.weights,inputs)) + self.bias
        if self.activation_function == "sigmoid": return self.sigmoid(raw_value)
        if self.activation_function == "tanh": return self.tanh(raw_value)
        if self.activation_function == "relu": return self.relu(raw_value)
        if self.activation_function == "step": return self.step(raw_value)
        





    def step(self,x:float) -> float:
        return 1 if x >= 0 else 0
    def tanh(self,x:float) -> float:
        exponential = np.exp(2*x) 
        return (exponential - 1)/(exponential + 1)
    def sigmoid(self,x:float) -> float:
        return 1/ (1 + exp(-x))
    def relu(self,x:float) -> float:
        return max(0,x)
    





        
    



        
node = perceptron(2,"step")


WEIGHTS = np.array([
    1,
    1
])

BIAS = -2.0


INPUTS = [
        1,
        1
    ]
node.set_weights(WEIGHTS)
node.set_bias(BIAS)
print(f"weights: {node.get_weights()}")
print(f"bias: {node.get_bias()}")


# generate the truth table for or function
print(node.calculate(INPUTS))