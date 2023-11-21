# this is a file that contains a class that has different types of optimizers 


import numpy as np
import fastNeuron.Activation_funcs as activations

class MiniBatchGradientDescent:

    """
    implemenation of the mini-batch gradient descent algorithm
    :param model: neural network object
    :param loss: loss function object
    :param X: training examples
    :param Y: training labels
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :param batch_size: size of each mini-batch
    :return the updated model and the costs of the model
    """
    def __call__(
                    self,
                    model:object,
                    loss:object,
                    X:np.ndarray,
                    Y:np.ndarray,
                    learning_rate:float,
                    epochs:int,
                    batch_size:int,
                    *args,
                    **kwargs
                ) -> tuple[object,list[float]]:
        
        
        # first shuffle the dataset to make sure it is random 
        # NOTE: we have to shuffle the labels accordingly

        shuffler = np.arange(X.shape[0])
        np.random.shuffle(shuffler)
        X = X[shuffler]
        Y = Y[shuffler]



        batch_number = np.floor(X.shape[0] / batch_size)
        costs = []

        mini_batches = np.array_split(X,batch_number)
        mini_batch_labels = np.array_split(Y,batch_number)
        # these variabels will be used for progress printing purposes
        if epochs < 100:
            cost_calculator = 1
            if epochs < 10:
                printer = 1
            else:
                printer = epochs//10
        elif epochs < 1000:
            cost_calculator = epochs//10
            printer = epochs//10
        else:
            cost_calculator = epochs//100
            printer = epochs//10

        for epoch in range(epochs):

            # each epoch go through all mini batches and update the weight and biases
            for mini_batch,label in zip(mini_batches,mini_batch_labels):
                # reset outputs
                model.outputs = []

                output = model.forward(mini_batch)

                # initialize derivatives that will be used to update layers one by one
                initial_derivatives = model.layers[-1].activation.derivative() * loss.derivative(output,label)


                # since this is called backpropagation we have to go backwards through the layers

                for index in range(len(model.layers) - 1 , -1 ,-1):



                    # update the values of the weights and biases

                    model.layers[index].backward(initial_derivatives,learning_rate)

                    # to go the next layer update the initial derivatives to contain the derivatives with respect to the inputs of that layer
                    # this is important so that we can update the weights of the previous layer
                    initial_derivatives = np.matmul(initial_derivatives,model.layers[index].weights.T)
                    # then we will multiply by the derivative of the activation function of the previous layer whatever it maybe
                    initial_derivatives *= model.layers[index-1].activation.derivative()

            if epoch % cost_calculator == 0:
                cost = loss(model.forward(X),Y)
                costs.append(cost)
            if epoch % printer == 0:
                accuracy = model.evaluate(model.predict(X),Y)
                print(f"{epoch} cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

        cost = loss(model.forward(X),Y)
        accuracy = model.evaluate(model.predict(X),Y)
        print(f" cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

        return model,costs

    def __repr__(self) -> str:
        return "Mini-Batch Gradient Descent"


class StochasticGradientDescent:
    """
    implemenation of the stochastic gradient descent algorithm
    :param model: neural network object
    :param loss: loss function object
    :param X: training examples
    :param Y: training labels
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :return the updated model and the costs of the model
    """
    def __call__(
                self,
                model:object, 
                loss:object,
                X:np.ndarray,
                Y:np.ndarray,
                learning_rate:float,
                epochs:int,
                *args,
                **kwargs
            ) -> tuple[object,list[np.float64]]:
        
        costs=[]



        # these variabels will be used for progress printing purposes
        if epochs < 100:
            cost_calculator = 1
            if epochs < 10:
                printer = 1
            else:
                printer = epochs//10
        elif epochs < 1000:
            cost_calculator = epochs//10
            printer = epochs//10
        else:
            cost_calculator = epochs//100
            printer = epochs//10

        # do as many epochs as needed
        for epoch in range(epochs):
            # go over all of the training examples in each epoch
            for i in range(X.shape[0]):
                model.outputs = []
                output = model.forward(X[i].reshape(1,X.shape[1]))
                initial_derivatives = model.layers[-1].activation.derivative() * loss.derivative(output,Y[i].reshape(1,Y.shape[1]))


                # since this is called backpropagation we have to go backwards through the layers
                for index in range(len(model.layers) - 1 , -1 ,-1):


                    # update the values of the weights and biases
                    model.layers[index].backward(initial_derivatives,learning_rate)

                    # to go the next layer update the initial derivatives to contain the derivatives with respect to the inputs of that layer
                    # this is important so that we can update the weights of the previous layer
                    initial_derivatives = np.matmul(initial_derivatives,model.layers[index].weights.T)
                    # then we will multiply by the derivative of the activation function of the previous layer whatever it maybe
                    initial_derivatives *= model.layers[index-1].activation.derivative()



            
            if epoch % cost_calculator == 0:
                cost = loss(model.forward(X),Y)
                costs.append(cost)
            if epoch % printer == 0:
                accuracy = model.evaluate(model.predict(X),Y)
                print(f"{epoch} cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")



        cost = loss(model.forward(X),Y)
        accuracy = model.evaluate(model.predict(X),Y)
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
    :return the updated model and the costs of the model
    """
    def __call__(
                    self, 
                    model:object, # neural network object
                    loss:object, #loss function object 
                    X:np.ndarray,
                    Y:np.ndarray,
                    learning_rate:float,
                    epochs:int,
                    *args,
                    **kwargs
                 ) -> tuple[object,list[np.float64]]:
        
        # NOTE: this algorithm can be implemented using recursion or using iterations
        #  i chose to use iterations for performance purposes...

        COSTS = []



        # these variabels will be used for progress printing purposes
        if epochs < 100:
            cost_calculator = 1
            if epochs < 10:
                printer = 1
            else:
                printer = epochs//10
        elif epochs < 1000:
            cost_calculator = epochs//10
            printer = epochs//10
        else:
            cost_calculator = epochs//100
            printer = epochs//10

        for epoch in range(epochs):


            # reset the outputs since they get updated on each forward pass
            model.outputs = []
            # this is the output of the neural network which is the last one in the list
            output = model.forward(X)

            if epoch % cost_calculator == 0:
                cost = loss(output,Y)
                COSTS.append(cost)
            if epoch % printer  == 0:
                accuracy = model.evaluate(model.predict(X),Y)
                print(f"{epoch} cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

            # get the initial derivatives depending ont he loss function
            initial_derivatives = loss.derivative(output,Y) * model.layers[-1].activation.derivative()
            
            # since this is called backpropagation we have to go backwards through the layers
            for index in range(len(model.layers) - 1 , -1 ,-1):


                # update the values of the weights and biases

                model.layers[index].backward(initial_derivatives,learning_rate)



                # to go the next layer update the initial derivatives to contain the derivatives with respect to the inputs of that layer
                # this is important so that we can update the weights of the previous layer
                initial_derivatives = np.matmul(initial_derivatives,model.layers[index].weights.T)
                # then we will multiply by the derivative of the activation function of the previous layer whatever it maybe
                initial_derivatives *= model.layers[index-1].activation.derivative()


        accuracy = model.evaluate(model.predict(X),Y)
        print(f" cost : {cost} accuracy : {round(accuracy,ndigits=2)}%")

        return model,COSTS
    

    def __repr__(self) -> str:
        return "Batch Gradient Descent"               
