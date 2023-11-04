# this is a file that contains a class that has different types of optimizers 


import numpy as np
class MiniBatchGradientDescent:
    pass

class StochasticGradientDescent:
    """
    implemenation of the stochastic gradient descent algorithm
    :param model: neural network object
    :param loss: loss function object
    :param X: training examples
    :param Y: training labels
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :return: the model and the costs of the model
    """
    def __call__(
                self,
                model:object, 
                loss:object,
                X:np.ndarray,
                Y:np.ndarray,
                learning_rate:float,
                epochs:int
            ) -> tuple[object:list[np.float64]]:
        
        costs=[]

        # do as many epochs as needed
        for epoch in range(epochs):
            # go over all of the training examples in each epoch
            for i in range(X.shape[0]):
                model.outputs = []
                output = model.forward(X[i].reshape(1,X.shape[1]))
                
                initial_derivatives = loss.initial_derivative(output,Y[i].reshape(1,1))

                for index in range(len(model.layers) - 1 , -1 ,-1):
                    model.layers[index].weights = model.layers[index].weights - learning_rate *  np.matmul(model.outputs[index].T,initial_derivatives)
                    model.layers[index].biases = model.layers[index].biases - learning_rate * np.sum(initial_derivatives , axis= 0)
                    initial_derivatives = np.matmul(initial_derivatives,model.layers[index].weights.T)
                    positive_values_derivative = (model.outputs[index] > 0).astype(int)
                    if model.layers[index].activation == "sigmoid": initial_derivatives *= model.outputs[index] * (1 - model.outputs[index])
                    elif model.layers[index].activation == "relu": initial_derivatives *= positive_values_derivative
                    elif model.layers[index].activation == "leaky-relu": initial_derivatives *= np.where(positive_values_derivative == 0,0.1,positive_values_derivative)
                    elif model.layers[index].activation == "tanh": initial_derivatives *= 1 - (model.outputs[index] ** 2)

            
            if epoch % (epochs // 100) == 0:
                cost = loss.calculate(model.forward(X),Y)
                costs.append(cost)
            if epoch % (epochs // 10) == 0:
                accuracy = ( (X.shape[0] - np.sum((model.predict(X) != Y).astype(int))) / X.shape[0] ) * 100
                print(f"{epoch} cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")



        cost = loss.calculate(model.forward(X),Y)
        accuracy = ( (X.shape[0] - np.sum((model.predict(X) != Y).astype(int))) / X.shape[0] ) * 100
        print(f" cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

        return model,costs
    
    def __repr__(self) -> str:
        return "Stochastic Gradient Descent"    


class BatchGradientDescent:
    """
    implementation of the batch gradient descent algorithm
    :param model: neural network object
    :param loss: loss function object
    :param X: training examples
    :param Y: training labels
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :return: the model and the costs of the model
    """
    def __call__(
                    self, 
                    model:object, # neural network object
                    loss:object, #loss function object 
                    X:np.ndarray,
                    Y:np.ndarray,
                    learning_rate:float,
                    epochs:int

                 ) -> tuple[object,list[np.float64]]:
        
        # NOTE: this algorithm can be implemented using recursion or using iterations
        #  i chose to use iterations for performance purposes...

        COSTS = []
        for epoch in range(epochs):
            # this is the output of the neural network which is the last one in the list
            output = model.forward(X)

            if epoch % (epochs // 100) == 0:
                cost = loss.calculate(output,Y)
                COSTS.append(cost)
            if epoch % (epochs // 10) == 0:
                accuracy = ((X.shape[0] - np.sum(((output >= 0.5).astype(int) != Y).astype(int))) / X.shape[0] ) * 100
                print(f"{epoch} cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

            # get the initial derivatives depending ont he loss function
            initial_derivatives = loss.initial_derivative(output,Y)
            
            # since this is called backpropagation we have to go backwards through the layers
            for index in range(len(model.layers) - 1 , -1 ,-1):


                # update the values of the weights

                # what we want to do it multiply each initial derivative with the corresponding input and then take the average
                # this will give us the same thing but faster
                model.layers[index].weights -= learning_rate * (1/ initial_derivatives.shape[0] * np.matmul(model.outputs[index].T,initial_derivatives))

                # we have to multiply the initial derivatives by 1 to get the bias gradient and then average so that's we are doing
                model.layers[index].biases -= learning_rate * (1/ initial_derivatives.shape[0] * np.sum(initial_derivatives , axis= 0))

                # layer_wgrads = 1/ initial_derivatives.shape[0] * np.matmul(model.outputs[index].T,initial_derivatives)
                # layer_bgrad = 1/ initial_derivatives.shape[0] * np.sum(initial_derivatives , axis= 0)
                # layer_igrads = initial_derivatives * model.layers[index].weights

                # to go the next layer update the initial derivatives to contain the derivatives with respect to the inputs of that layer
                # this is important so that we can update the weights of the previous layer
                initial_derivatives = np.matmul(initial_derivatives,model.layers[index].weights.T)
                # then we will multiply by the derivative of the activation function of the previous layer whatever it maybe
                if model.layers[index].activation == "sigmoid": initial_derivatives *= model.outputs[index] * (1 - model.outputs[index])
                elif model.layers[index].activation == "relu": initial_derivatives *= (model.outputs[index] > 0).astype(int)
                elif model.layers[index].activation == "tanh": initial_derivatives *= 1 - (model.outputs[index] ** 2)

        accuracy = ((X.shape[0] - np.sum(((output >= 0.5).astype(int) != Y).astype(int))) / X.shape[0] ) * 100
        print(f" cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

        return model,COSTS
    

    def __repr__(self) -> str:
        return "Batch Gradient Descent"               
