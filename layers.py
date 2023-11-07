# this file will be used to create the layer class

# a layer in a neural network is a data structure that holds a bunch of perceptrons(neurons)
# it is a higher level abstraction to abstract some of the code we have to write to propogate forward and backward through the network
# in other words it is a container for the perceptrons

# import dependencies
import numpy as np




class LayerNorm:
    """
    a layer normalization layer that takes the outputs of the previous layer and normalize them sample by sample
    """
    def __init__(
                    self,
                    input_size:int,
                    epsilon:float = 1e-6,
                    gamma:np.ndarray = None,
                    beta:np.ndarray = None,
                
                ) -> None:
        self.input_size = input_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
    


    def forward(self,X:np.ndarray):
        """
        forward pass of the layer

        we will calculate the mean and variance of each input sample and normalize each sample and return the normalized samples
        X: input data
        returns: normalized data
        """
        mean = np.mean(X,axis=0,keepdims=True)
        std = np.std(X,axis=0,keepdims=True)

        # for now i will not use gamma and beta will look into them later
        return (X - mean) / (std + self.epsilon)

class Dense:
    """
    Dense layer or in other words fully connected layer where each input is connected to all of the outputs
    """

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
        assert activation in ["sigmoid","tanh","relu","linear","leaky-relu","softmax"],"activation function must be one of the following: sigmoid,tanh,relu,linear,softmax"


        # save values to the object
        self.input_size = input_size
        self.units = units
        self.activation = activation
        self.inputs = None

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
        """
        sigmoid activation function
        z: input value
        returns: sigmoid activation value of z
        """
        return 1/(1+np.exp(-z))
    @staticmethod
    def tanh(z):
        """
        tanh activation function
        z: input value
        returns: tanh activation value of z
        """
        return np.tanh(z)
    @staticmethod
    def relu(z):
        """
        relu activation function
        z: input value
        returns: relu activation value of z
        """
        return np.maximum(0,z)
    @staticmethod
    def leaky_relu(z):
        """
        leaky relu activation function
        z: input value
        returns: leaky relu activation value of z
        """
        return np.maximum(0.1*z,z)
    @staticmethod
    def linear(z):
        """
        linear activation function (no activation)
        z: input value
        returns: linear activation value of z
        """
        return z
    @staticmethod
    def softmax(z):
        """
        softmax activation function - usually used for the last layer as a probability distribution in classification
        z: input value
        returns: softmax activation value of z
        """
        return np.exp(z)/np.sum(np.exp(z))

    # forward propagation
    def forward(self,X:np.ndarray) -> np.ndarray:
        self.inputs = X
        raw = X @ self.weights + self.biases
        if self.activation == "sigmoid": return Dense.sigmoid(raw)
        elif self.activation == "tanh": return Dense.tanh(raw)
        elif self.activation == "relu": return Dense.relu(raw)
        elif self.activation == "leaky-relu": return Dense.leaky_relu(raw)
        elif self.activation == "linear": return Dense.linear(raw)
        elif self.activation == "softmax": return Dense.softmax(raw)


    def backward(
                    self,
                    initial_derivatives:np.ndarray,
                    learning_rate:float,
                 ) -> None:
        """
        backpropogation for this layer
        i implemented it this way so that we can implement different layer types and always call .backward()
        """


        # update the values of the weights

        # what we want to do it multiply each initial derivative with the corresponding input and then take the average
        # this will give us the same thing but faster
        self.weights = self.weights - learning_rate * np.matmul(self.inputs.T,initial_derivatives)

        # we have to multiply the initial derivatives by 1 to get the bias gradient and then average so that's we are doing
        self.biases = self.biases - learning_rate * np.sum(initial_derivatives,axis=0)

        
