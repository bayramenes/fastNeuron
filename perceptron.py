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
                    self,input_number:int,
                    # which layer is perceptron located at (indexing start from 0)
                    activation_function:str
                 ) -> None:
        

        self.input_number = input_number
        self.activation_function = activation_function
        # initialize random values for weights and biases
        self.weights = randint(5,10,size=(self.input_number,))
        self.bias = randint(5,10)

        # store some values so that we can compute derivatives and so on faster
        # all values will be initialized to 0
        self.inputs = None
        self.raw_output = None
        self.output = None


    def __repr__(self) -> str:
        return f"perceptron with {self.inputs} inputs and {self.activation_function} activation function"
    
    # implementation of the most famous activation functionos
    # NOTE: RELU is implement as "leaky" RELU by default
    def tanh(self,x:float) -> float:
        exponential = np.exp(2*x) 
        return (exponential - 1)/(exponential + 1)
    def tanh_derivative(self) -> float:
        return 1 - self.output**2
    def sigmoid(self,x:float) -> float:
        return 1/ (1 + exp(-x))
    def sigmoid_derivative(self) -> float:
        return self.output*(1-self.output)
    # by default i implemented leaky RELU to avoid neurons from dying
    def relu(self,x:float) -> float:
        return x if x > 0 else 0.1 * x
    # we will return 0 for the derivative at 0 though it is mathematically not differentiable at 0 but in practice that works
    def relu_derivative(self,x:float) -> float:
        return 1 if x > 0 else 0.1
    

    # when calculating derivatives for backpropagation this function will be used
    def output_derivative(self) -> float:
        if self.activation_function == "sigmoid": return self.sigmoid_derivative()
        if self.activation_function == "tanh": return self.tanh_derivative()
        if self.activation_function == "relu": return self.relu_derivative()
        if self.activation_function == "none": return 1
    

    def forward(self,inputs:list[float]) -> float :
        # make sure that input is the correct size and type
        # assert len(inputs) == self.input_number,"inputs must be of the same size as the number of inputs"

        # make sure that inputs is a numpy array 
        if not isinstance(inputs,ndarray):
            inputs = np.array(inputs)
        # set the inputs for the object as these ones given
        self.inputs = inputs

        # we now calculate the value of the perceptron
        # it is the dot product of the inputs and weights
        # set the raw output of the perceptron
        # this will be useful when finding the derivative 
        self.raw_output = np.dot(self.weights,inputs) + self.bias

        # use the appropriate activation function
        if self.activation_function == "sigmoid": self.output = self.sigmoid(self.raw_output)
        elif self.activation_function == "tanh": self.output = self.tanh(self.raw_output)
        elif self.activation_function == "relu": self.output = self.relu(self.raw_output)
        elif self.activation_function == "none": self.output = self.raw_output
        return self.output



    def weight_grads(self,previous_derivatives:float) -> ndarray:
        return previous_derivatives * self.output_derivative() * self.inputs
    
    def bias_grad(self,previous_derivatives:float) -> float:
        return previous_derivatives * self.output_derivative() * 1
    

    def input_grads(self,previous_derivatives:float) -> ndarray:
        return previous_derivatives * self.output_derivative() * self.weights
    
    # def grads(self,previous_derivatives:(float,int)) -> ndarray:
    #     # previous derivatives is the multiplication of the chain rule up until this point
    #     # next we will multiply the previous derivatives with the derivative of the activation function
    #     # then we will multiply that with the inputs to the gradient since the derivative of w_i is x_i

    #     activation_derivative = previous_derivatives * self.output_derivative()

    #     # return (weight grads , bias gradient ,input grads)
    #     return  activation_derivative * self.inputs , activation_derivative * 1 , activation_derivative * self.weights
