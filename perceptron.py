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

from numpy.random import uniform,randint
from numpy import ndarray,exp
import numpy as np



# create a perceptron class

class perceptron:
    def __init__(
                    self,input_number:int,
                    # which is this perceptron located
                    layer:int,
                    activation_function:str
                 ) -> None:
        

        # make sure that the values given are correct
        assert isinstance(input_number,int),"input_number must be an integer represeting the number of input_number"
        assert isinstance(layer,int),"layer must be an integer represeting which layer is this perceptron at"
        assert isinstance(activation_function,str),"activation function must be a string"
        assert activation_function in ["sigmoid","tanh","relu","none"],"activation function must be one of the following: sigmoid,tanh,relu,step"

        self.input_number = input_number
        self.activation_function = activation_function
        # initialize random values for weights and biases
        self.weights = randint(5,10,size=(self.input_number,))
        self.bias = randint(5,10)
        self.layer = layer
        self.inputs = None
        self.raw_output = None


    def __repr__(self) -> str:
        return f"perceptron with {self.inputs} inputs and {self.activation_function} activation function"
    

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
    
    def output_derivative(self) -> float:
        if self.activation_function == "sigmoid": return self.sigmoid_derivative()
        if self.activation_function == "tanh": return self.tanh_derivative()
        if self.activation_function == "relu": return self.relu_derivative()
        if self.activation_function == "none": return 1
    
    def forward(self,inputs:list[float]) -> float :
        # make sure that input is the correct size and type
        assert len(inputs) == self.input_number,"inputs must be of the same size as the number of inputs"

        if not isinstance(inputs,ndarray):
            inputs = np.array(inputs)
        # set the inputs for the object as these ones
        self.inputs = inputs

        # we now calculate the value of the perceptron
        # it is the dot product of the inputs and weights
        raw_value = np.dot(self.weights,inputs) + self.bias

        # set the raw output of the perceptron
        # this will be useful when finding the derivative 
        self.raw_output = raw_value

        # use the appropriate activation function
        if self.activation_function == "sigmoid": self.output = self.sigmoid(raw_value)
        elif self.activation_function == "tanh": self.output = self.tanh(raw_value)
        elif self.activation_function == "relu": self.output = self.relu(raw_value)
        elif self.activation_function == "none": self.ouput = raw_value
        return self.output


    def backward(self,previous_derivatives:(float,int)) -> ndarray:
        # previous derivatives is the multiplication of the chain rule up until this point
        # next we will multiply the previous derivatives with the derivative of the activation function
        # then we will multiply that with the inputs to the gradient since the derivative of w_i is x_i

        # return (weight grads , bias gradient)
        return previous_derivatives * self.output_derivative() * self.inputs , previous_derivatives * self.output_derivative() * 1
