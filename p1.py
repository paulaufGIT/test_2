
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1.2, 5.2, 8, 2],[8, 8, 8, 2],[1, 8, 7, 0]]

X, Y = spiral_data(200, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)