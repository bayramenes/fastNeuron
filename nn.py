import numpy as np
from perceptron import perceptron



node = perceptron(2,"step")


WEIGHTS = np.array([
    1,
    1
])

BIAS = -2.0


INPUTS = [
        0,
        0
    ]
node.set_weights(WEIGHTS)
node.set_bias(BIAS)
print(f"weights: {node.get_weights()}")
print(f"bias: {node.get_bias()}")


# generate the truth table for or function
print(node.calculate(INPUTS))