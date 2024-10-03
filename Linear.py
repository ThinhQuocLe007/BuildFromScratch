import numpy as np
from module import Module 
class Linear(Module):
    def __init__(self, n_features, n_neurons):
        """
        Initializes a linear (fully connected) layer.

        Constraints:
            - input.shape = (m, n)  # (batch_size, n_features)
            - W.shape = (n, o)      # (n_features, n_neurons)
            - b.shape = (o,)        # (n_neurons,) -> broadcasted to (m, o)
            - output.shape = (m, o) # (batch_size, n_neurons)

        Parameters:
            - n_features (int): The number of input features
            - n_neurons (int): The number of neurons in this layer
        """
        # He initialization for weights and zeros for biases
        self.params = {
            'W': np.random.randn(n_features, n_neurons) * np.sqrt(2. / n_features),
            'b': np.zeros(n_neurons)
        }
        # Initialize gradients as empty dictionaries
        self.grads = {
            'dW': np.zeros_like(self.params['W']),
            'db': np.zeros_like(self.params['b'])
        }

    def forward(self, input):
        """
        Performs the forward pass of the linear layer.

        Parameters:
            - input (np.array): The input array to the layer

        Returns:
            - output (np.array): The result of the linear transformation
        """
        self.X = input  # Save the input for backward pass
        W = self.params['W']
        b = self.params['b']
        # Linear transformation
        output = np.dot(self.X, W) + b
        return output

    def backward(self, dout):
        """
        Performs the backward pass and calculates the gradients.

        Parameters:
            - dout (np.array): The gradient of the next layer (n + 1)

        Returns:
            - dx (np.array): The gradient with respect to the input (n - 1)
        """
        # Calculate gradients
        self.grads['dW'] = np.dot(self.X.T, dout)  # Gradient of weights
        self.grads['db'] = np.sum(dout, axis=0)    # Gradient of biases
        # Gradient with respect to the input to propagate to the previous layer
        dx = np.dot(dout, self.params['W'].T)
        return dx
