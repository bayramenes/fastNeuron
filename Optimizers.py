# this is a file that contains a class that has different types of optimizers 

from layer import layer
from network import neural_network
import numpy as np
import loss_funcs


class Optimizers:


    @staticmethod
    def BatchGradientDescent(model:neural_network,loss,X:np.ndarray,Y:np.ndarray,learning_rate:float,epochs:int) -> tuple[neural_network,list[np.float64]]:


        # NOTE: this algorithm can be implemented using recursion or using iterations
        #  i chose to use iterations for performance purposes...

        COSTS = []
        for epoch in epochs:
            # this is the output of the neural network which is the last one in the list
            output = model.forward(X)


            if epoch % (epochs//10) == 0:
                cost = loss.calculate(output,Y)
                print(f"epoch {epoch} cost: {cost}")
            if epoch % (epochs//100) == 0:
                COSTS.append(cost)

            # get the initial derivatives depending ont he loss function
            initial_derivatives = loss.initial_derivative(output,Y)
            
            # since this is called backpropagation we have to go backwards through the layers
            for index in range(len(model.layers) - 1 , -1 ,-1):


                # update the values of the weights
                model.layers[index].weights -= learning_rate * (1/ initial_derivatives.shape[0] * np.matmul(model.outputs[index].T,initial_derivatives))
                model.layers[index].bias -= learning_rate * (1/ initial_derivatives.shape[0] * np.sum(initial_derivatives , axis= 0))

                # layer_wgrads = 1/ initial_derivatives.shape[0] * np.matmul(model.outputs[index].T,initial_derivatives)
                # layer_bgrad = 1/ initial_derivatives.shape[0] * np.sum(initial_derivatives , axis= 0)
                # layer_igrads = initial_derivatives * model.layers[index].weights
                # for now i am assuming that we are only going to use the sigmoid function as an activation function

                # to go the next layer update the initial derivatives to contain the derivatives with respect to the inputs of that layer
                # this is important so that we can update the weights of the previous layer
                # then we will multiply by the derivative of the activation function of the previous layer whatever it maybe

                initial_derivatives = np.matmul(initial_derivatives,model.layers[index].weights.T)
                if model.layers[index].activation == "sigmoid": initial_derivatives *= model.outputs[index] * (1 - model.outputs[index])
                elif model.layers[index].activation == "relu": initial_derivatives *= (model.outputs[index] > 0).astype(int)
                elif model.layers[index].activation == "tanh": initial_derivatives *= 1 - (model.outputs[index] ** 2)


        return model,COSTS
                
                




    def MiniBatchGradientDescent(X:np.ndarray,Y:np.ndarray,learning_rate:float,epochs:int,batches:int):
        pass

    def StochasticGradientDescent(X:np.ndarray,Y:np.ndarray,learning_rate:float,epochs:int):
        pass