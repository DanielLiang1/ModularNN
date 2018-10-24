import numpy as np


"""
This file implements activation layers of different types
in line with a computational graph model
"""

class ReluLayer:

    def __init__(self, shape):
        """

        :param shape: shape of input to the layer
        """
        self.A = np.zeros(shape)  # create variable to store activations

    def activate(self, Z):
        """

        :param Z: input from previous (linear) layer
        """
        self.A = np.maximum(0, Z)  # ReLU activation function

    def backward(self, upstream_grad):
        """

        Relu backward pass is like routing upstream gradients to non-zero activations

        :param upstream_grad: gradient coming into this layer from the layer above
        """
        self.dZ = upstream_grad
        self.dZ[self.A <= 0] = 0  # basically routing gradients by setting ones with zeros activation to zero


class SigmoidLayer:
    def __init__(self, shape):
        """

        :param shape: shape of input to the layer
        """
        self.A = np.zeros(shape)

    def activate(self, Z):
        """

        :param Z: input from previous (linear) layer
        """
        self.A = 1 / (1 + np.exp(-Z))

    def backward(self, upstream_grad):
        """
        derivative of sigmoid(local gradient) => A*(1-A)

        :param upstream_grad: gradient coming into this layer from the layer above
        """
        self.dZ = upstream_grad * self.A*(1-self.A)  # couple upstream gradient with local gradient
