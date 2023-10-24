from pprint import pprint
import numpy as np
from perceptron import perceptron
from sklearn import datasets


node = perceptron(2,0,"sigmoid")




# # generate random data
DATA, LABELS = datasets.make_classification(n_samples = 1000
                           ,n_features = 2
                           ,n_informative = 2
                           ,n_redundant = 0
                           ,n_clusters_per_class = 1
                           ,flip_y = 0
                           ,class_sep = 2
                           ,random_state = 7
                           )

def gradient_descent(data:list[list[(int,float)]],labels:list[(int,float)],learning_rate:float = 0.03 , epochs:int = 1000):
    # go through each epoch

    for epoch in range(epochs):

        # feed the data to the perceptron and get the output value

        weight_gradients = np.zeros(shape=node.weights.shape)
        bias_gradient = 0
        average_error = 0

        for data_sample_index in range(len(data)):

            # feed the data sample to the perceptron
            output = node.forward(data[data_sample_index])



            # calculate the error
            # this is called L2 loss
            error = (output - labels[data_sample_index]) ** 2
        


            # calculate the gradient of the derivatives to pass to the weight_gradients function
            # d(cost) / d(raw_output) = d(cost) / d(output) * d(output) / d(raw_output)
            # output is the raw_output after being passed through an activation whatever that may be tanh, sigmoid , relu
            previous_derivatives = 2 * (output - labels[data_sample_index])
            w_g,b_g = node.backward(previous_derivatives)


            weight_gradients = weight_gradients + w_g
            bias_gradient = bias_gradient + b_g

            # add the average error
            average_error += error

        # normalize the gradients
        # weight_gradients = weight_gradients / len(data)
        # bias_gradient = bias_gradient / len(data)

        # this is a matrix subtraction
        node.weights = node.weights - learning_rate * weight_gradients
        node.bias = node.bias - learning_rate * bias_gradient
        if epoch % 100 == 0:
            print(f"{epoch} : loss :{average_error}")
            # print("weights", node.weights)
            # print("bias", node.bias)

            print('-'* 50)
            print("\n")
    





def test():
    correct = 0
    outputs = [node.forward(data) for data in DATA]
    for output, label in zip(outputs, LABELS):
        if abs(output - label) < 0.01:
            correct += 1

    print(f"accuracy : {correct * 100 / len(DATA)}%")



if __name__ == "__main__":
    gradient_descent(DATA,LABELS)
    test()







