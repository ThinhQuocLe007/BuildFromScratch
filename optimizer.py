import numpy as np 

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError('The step method must be implemented by the subclass')


class SGD(Optimizer):
    def __init__(self, model, learning_rate=0.1, momentum=0.0):
        super().__init__(learning_rate)
        self.layers = model
        self.lr = learning_rate
        self.momentum = momentum
        # Initialize velocities for all parameters as a dictionary of layer IDs
        self.velocities = {id(layer): {param_name: 0 for param_name in layer.params} for layer in self.layers}

    def step(self):
        """
        Performs a single optimization step, updating the provided parameters.
        """
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for param_name in layer.params:
                    # Extract gradient
                    dparam = layer.grads[f'd{param_name}']

                    # Compute velocities
                    self.velocities[id(layer)][param_name] = self.momentum * self.velocities[id(layer)][param_name] - self.lr * dparam

                    # Update params together
                    layer.params[param_name] += self.velocities[id(layer)][param_name]


class Adam(Optimizer):
    def __init__(self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.layers = model
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step counter

        # Initialize first and second moment estimates for all parameters
        self.m = {id(layer): {param_name: 0 for param_name in layer.params} for layer in self.layers if hasattr(layer, 'params')}
        self.v = {id(layer): {param_name: 0 for param_name in layer.params} for layer in self.layers if hasattr(layer, 'params')}

    def step(self):
        """
        Performs a single optimization step using the Adam algorithm.
        """
        self.t += 1  # Increment time step
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for param_name in layer.params:
                    # Extract gradient
                    dparam = layer.grads[f'd{param_name}']

                    # Update first moment estimate
                    self.m[id(layer)][param_name] = self.beta1 * self.m[id(layer)][param_name] + (1 - self.beta1) * dparam

                    # Update second moment estimate
                    self.v[id(layer)][param_name] = self.beta2 * self.v[id(layer)][param_name] + (1 - self.beta2) * (dparam ** 2)

                    # Correct bias in moment estimates
                    m_hat = self.m[id(layer)][param_name] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[id(layer)][param_name] / (1 - self.beta2 ** self.t)

                    # Update parameters together
                    layer.params[param_name] -= self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)
