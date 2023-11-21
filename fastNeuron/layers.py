# this file will be used to create the layer class

# a layer in a neural network is a data structure that holds a bunch of perceptrons(neurons)
# it is a higher level abstraction to abstract some of the code we have to write to propogate forward and backward through the network
# in other words it is a container for the perceptrons

# import dependencies
import numpy as np
import fastNeuron.Activation_funcs as activations




class LayerNorm:
    """
    a layer normalization layer that takes the outputs of the previous layer and normalize them sample by sample
    """
    def __init__(
                    self,
                    input_size:int,
                    epsilon:float = 1e-6,
                
                ) -> None:
        self.input_size = input_size
        self.epsilon = epsilon
        self.gamma = np.random.randn(1,input_size) 
        self.beta = np.random.randn(1,input_size)
        self.inputs = None
        self.outputs = None



    


    def forward(self,X:np.ndarray):
        """
        forward pass of the layer

        we will calculate the mean and variance of each input sample and normalize each sample and return the normalized samples
        X: input data
        returns: normalized data
        """

        self.inputs = X
        mean = np.mean(X,axis=0,keepdims=True)
        std = np.std(X,axis=0,keepdims=True)
        self.outputs = (X - mean) / (std + self.epsilon)
        # for now i will not use gamma and beta will look into them later
        return self.gamma * self.outputs + self.beta
    


    def backward(self,initial_derivatives:np.float64,learning_rate:np.float64) -> np.float64:
        """
        backward pass of the layer

        we will calculate the derivatives of the layer and return the derivatives
        initial_derivatives: derivatives of the layer
        learning_rate: learning rate of the layer
        returns: derivatives of the layer
        """
        
        # calculate the derivatives of the layer
        derivatives = (initial_derivatives * self.outputs) / self.inputs.shape[0]

        # return the derivatives
        return derivatives

class Dense:
    """
    Dense layer or in other words fully connected layer where each input is connected to all of the outputs
    """

    # initialize the layer
    def __init__(
                self,
                input_size:int,
                units:int,
                activation:object,
            ) -> None:
        

        # make sure values are correct
        assert isinstance(input_size,int),"input_number must be an integer represeting the number of input_number"
        
        


        # save values to the object
        self.input_size = input_size
        self.units = units
        self.activation = activation
        self.inputs = None

        # initialize the weight of the perceptrons according to Xavier/Glorot Initialization
        sd = np.sqrt(1/(input_size + units))
        if isinstance(activation,activations.sigmoid): sd *= np.sqrt(6)
        elif isinstance(activations,(activations.relu)): sd *= np.sqrt(2)


        # a matrix that has all of the values of the weights of all of the units
        self.weights = np.random.normal(0,sd,size=(input_size,units))
        self.biases = np.random.normal(0,sd,size=(1,units))

    def __repr__(self) -> str:
        return f"layer with {self.units} units, {self.input_size} inputs and {self.activation} activation function"

    # forward propagation
    def forward(self,X:np.ndarray) -> np.ndarray:
        self.inputs = X
        raw = X @ self.weights + self.biases
        return self.activation(raw)


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
        self.weights = self.weights - learning_rate  * np.matmul(self.inputs.T,initial_derivatives)

        # we have to multiply the initial derivatives by 1 to get the bias gradient and then average so that's we are doing
        self.biases = self.biases - learning_rate * np.sum(initial_derivatives,axis=0)



        
