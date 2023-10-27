from pprint import pprint
import numpy as np
from perceptron import perceptron
from sklearn import datasets
from layer import layer
from network import neural_network
import matplotlib.pyplot as plt




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

# # generate random data
DATA, LABELS = datasets.make_circles(
                                n_samples = 300,
                                noise=0.03
                           )

LABELS = np.reshape(LABELS,newshape=(LABELS.shape[0],1))
M = DATA.shape[0]
N = DATA.shape[1]

# create a model
model  = neural_network()
model  = model.sequential(
    [
        layer(2,20,"relu"),
        # layer(20,10,"sigmoid"),
        layer(20,1,"sigmoid")
    ]
)

# train model
model,costs = model.fit(DATA,LABELS,0.03,10000)

# plot cost
plt.plot(costs)


# plot decision boundary
h = 0.02
x_min , x_max = DATA[:,0].min() - 1 , DATA[:,0].max() + 1
y_min , y_max = DATA[:,1].min() - 1 , DATA[:,1].max() + 1
xx , yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
np.linspace(y_min,y_max,100)
x_in = np.c_[xx.ravel(),yy.ravel()]
y_pred = model.predict(x_in).reshape(xx.shape)

plt.contourf(xx,yy,y_pred,cmap= plt.cm.RdYlBu)
plt.scatter(DATA[:,0],DATA[:,1],c = LABELS)






# def gradient_descent(data:list[list[(int,float)]],labels:list[(int,float)],learning_rate:float = 0.03 , epochs:int = 1000):
#     # go through each epoch

#     for epoch in range(epochs):

#         # feed the data to the perceptron and get the output value

#         weight_gradients = np.zeros(shape=node.weights.shape)
#         bias_gradient = 0
#         average_error = 0

#         for data_sample_index in range(len(data)):

#             # feed the data sample to the perceptron
#             output = node.forward(data[data_sample_index])



#             # calculate the error
#             # this is called L2 loss
#             error = (output - labels[data_sample_index]) ** 2
        


#             # calculate the gradient of the derivatives to pass to the weight_gradients function
#             # d(cost) / d(raw_output) = d(cost) / d(output) * d(output) / d(raw_output)
#             # output is the raw_output after being passed through an activation whatever that may be tanh, sigmoid , relu
#             previous_derivatives = 2 * (output - labels[data_sample_index])
#             w_g,b_g = node.grads(previous_derivatives)


#             weight_gradients = weight_gradients + w_g
#             bias_gradient = bias_gradient + b_g

#             # add the average error
#             average_error += error

#         # normalize the gradients
#         # weight_gradients = weight_gradients / len(data)
#         # bias_gradient = bias_gradient / len(data)

#         # this is a matrix subtraction
#         node.weights = node.weights - learning_rate * weight_gradients
#         node.bias = node.bias - learning_rate * bias_gradient
#         if epoch % 100 == 0:
#             print(f"{epoch} : loss :{average_error}")
#             # print("weights", node.weights)
#             # print("bias", node.bias)

#             print('-'* 50)
#             print("\n")
    





# def test():
#     correct = 0
#     outputs = [node.forward(data) for data in DATA]
#     for output, label in zip(outputs, LABELS):
#         if abs(output - label) < 0.01:
#             correct += 1

#     print(f"accuracy : {correct * 100 / len(DATA)}%")





