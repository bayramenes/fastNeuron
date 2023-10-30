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
            layer_wgrads = 1/ initial_derivatives.shape[0] * np.matmul(self.outputs[index].T,initial_derivatives)
            layer_bgrads = 1/ initial_derivatives.shape[0] * np.sum(initial_derivatives , axis= 0)
            # layer_igrads = initial_derivatives * self.layers[index].weights
            # for now i am assuming that we are only going to use the sigmoid function as an activation function
            initial_derivatives = np.matmul(initial_derivatives,self.layers[index].weights.T) * (self.outputs[index] * (1 - self.outputs[index]))
            if self.layers[index].activation == "sigmoid": initial_derivatives *= self.outputs[index] * (1 - self.outputs[index])
            elif self.layers[index].activation == "relu": initial_derivatives *= (self.outputs[index] > 0).astype(int)
            elif self.layers[index].activation == "tanh": initial_derivatives *= 1 - self.outputs[index] * self.outputs[index]
                
            wgrads = [layer_wgrads] + wgrads
            bgrads = [layer_bgrads] + bgrads
            



        return wgrads , bgrads
    


    def fit(self,X:np.ndarray,Y:np.ndarray,learning_rate:float,epochs:int):

        costs=[]

        M = X.shape[0]
        N = X.shape[1]
        print(f"M : {M}")
        print(f"M//10 : {epochs//10}")
        print(f"M//100 : {epochs//100}")
        for epoch in range(epochs):
            outputs = self.forward(X)


            # binary cross-entropy

            if epoch % (epochs // 100) == 0:
                cost = -1 / M * np.sum(Y * np.log(outputs) + (1 - Y) * np.log(1 - outputs))
                costs.append(cost)
            if epoch % (epochs // 10) == 0:
                accuracy = ((M - np.sum(((outputs >= 0.5).astype(int) != Y).astype(int))) / M ) * 100
                print(f"{epoch} cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

            # get the gradients
            wgrads , bgrads = self.backward(outputs - Y)
            # update the weights and biases
            for index,layer in enumerate(self.layers):
                layer.weights -= learning_rate * wgrads[index]
                layer.biases -= learning_rate * bgrads[index]
        accuracy = ((M - np.sum(((outputs >= 0.5).astype(int) != Y).astype(int))) / M ) * 100
        print(f" cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")




        return self,costs
    


    def predict(self,X:np.ndarray):
        return (self.forward(X) >= 0.5).astype(int)
    


            






    


        
        