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