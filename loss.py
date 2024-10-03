import numpy as np 
from module import Module

class Loss(Module):    
    def __init__(self, loss_name, reduction='mean'):
        """
        Choose loss function from the supported list.
        Parameters:
            - loss_name (str): Name of the loss function. Options: [mse, binary_cross_entropy, categorical_cross_entropy, hinge]
            - reduction (str): Reduction type: [mean, sum, none]
        
        Return:
            - None
        """
        self.losses_func = {
            'mse': (self.mse, self.mse_prime),
            'binary_cross_entropy': (self.binary_cross_entropy, self.binary_cross_entropy_prime),
            'categorical_cross_entropy': (self.categorical_cross_entropy, self.categorical_cross_entropy_prime),
            'hinge': (self.hinge_loss, self.hinge_prime)
        }

        self.loss_name = loss_name.lower()
        self.reduction = reduction.lower()
        if self.loss_name not in self.losses_func:
            raise ValueError(f'Loss function {self.loss_name} is not supported. Choose from {list(self.losses_func.keys())}')

    def __apply_reduction(self, loss):
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'The reduction method {self.reduction} is not supported')

    def forward(self, y_true, y_pred):
        """
        Apply loss function.
        Parameters:
            - y_true (np.array): True values
            - y_pred (np.array): Predicted values
        
        Return:
            - Loss value after reduction
        """
        loss_function, _ = self.losses_func[self.loss_name]
        loss = loss_function(y_true, y_pred)
        return self.__apply_reduction(loss)

    def backward(self, y_true, y_pred):
        """
        Compute gradient of the loss.
        Parameters:
            - y_true (np.array): True values
            - y_pred (np.array): Predicted values
        
        Return:
            - Gradient for the previous layer
        """
        _, gradient = self.losses_func[self.loss_name]
        grad = gradient(y_true, y_pred)
        return grad

    @staticmethod
    def mse(y_true, y_pred):
        return (y_true - y_pred) ** 2

    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    # NEED TO UPDATE LIKE LEASON !!!!!!! 
    @staticmethod  
    def categorical_cross_entropy(y_true, y_pred):
        # Clip y_pred to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def categorical_cross_entropy_prime(y_true, y_pred):
        # For softmax combined with cross-entropy, the gradient simplifies to this
        return (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def hinge_loss(y_true, y_pred):
        return np.maximum(0, 1 - y_true * y_pred)

    @staticmethod
    def hinge_prime(y_true, y_pred):
        return np.where(1 - y_true * y_pred > 0, -y_true, 0)