from module import Module 
import numpy as np 

class Activation(Module):  
    def __init__(self, activation_name): 
        """
        Use to select activation function.
        
        Parameters:
            - activation_name (str): name of activation functi"o"n.
              Supported list: ['relu', 'leakyrelu', 'tanh', 'sigmoid', 'softmax']
        
        Return: 
            - None (apply activation function selected)
        """
        self.activation_functions = {
            'tanh': (self.tanh, self.tanh_prime), 
            'sigmoid': (self.sigmoid, self.sigmoid_prime), 
            'relu': (self.relu, self.relu_prime), 
            'leakyrelu': (self.leakyrelu, self.leakyrelu_prime), 
            'softmax': (self.softmax, self.softmax_prime)
        }

        self.activation_name = activation_name.lower() 
        if self.activation_name not in self.activation_functions: 
            raise ValueError(f'Activation function {self.activation_name} is not supported. Please choose from {list(self.activation_functions.keys())}')

    def forward(self, x): 
        """
        Apply activation function.
        
        Parameters: 
            - x (np.array): input array 
        
        Return: 
            - result (np.array) after applying the activation function.
        """
        self.x = x 
        activation_func, _ = self.activation_functions[self.activation_name]
        return activation_func(self.x) 

    def backward(self, output_gradient):
        """
        Compute gradient for the previous layer.
        
        Parameters: 
            - output_gradient (np.array): the gradient of the later layer.
        
        Return: 
            - input_gradient (np.array): gradient for the previous layer.
        """
        if self.activation_name == 'softmax': 
            # For softmax, usually combined with categorical cross-entropy, just return the output gradient
            return output_gradient
        else:
            _, gradient = self.activation_functions[self.activation_name]
            return output_gradient * gradient(self.x) 
    
    @staticmethod
    def sigmoid(x): 
        x = np.clip(x, -500, 500) 
        return 1 / (1 + np.exp(-x)) 
    
    @staticmethod
    def sigmoid_prime(x): 
        s = Activation.sigmoid(x)  
        return s * (1 - s)

    @staticmethod
    def tanh(x): 
        return np.tanh(x) 
    
    @staticmethod
    def tanh_prime(x): 
        return 1 - np.tanh(x) ** 2 
    
    @staticmethod
    def relu(x): 
        return np.maximum(0, x) 
    
    @staticmethod
    def relu_prime(x): 
        return np.where(x > 0, 1, 0) 

    @staticmethod
    def leakyrelu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x) 
    
    @staticmethod
    def leakyrelu_prime(x, alpha=0.1):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x): 
        # Stable softmax implementation
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return output 
    
    @staticmethod
    def softmax_prime(softmax_output): 
        """
        The derivative of softmax is usually not implemented separately because it is rarely used
        directly. When combined with cross-entropy loss, the gradient simplifies to just the 
        output gradient from the loss function, making it computationally eficient.
        
        If needed, the derivative involves computing a Jacobian matrix which can be complex 
        and is generally avoided in practice.
        
        Raise NotImplementedError to indicate its limited practical usage.
        """
        raise NotImplementedError(
            "The derivative of softmax is rarely used directly. In practice, "
            "softmax is combined with cross-entropy loss, which simplifies the "
            "gradient computation during backpropagation. If you need the exact "
            "Jacobian matrix, consider implementing a specialized function."
        )