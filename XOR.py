# Import libraries 
import numpy as np 
from Linear import Linear 
from optimizer import SGD 
from utils import DataLoader,OnehotEncoder 
from loss import Loss 
from activations import Activation
from module import Sequential

# Data 
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0]).reshape((4, 1))

# Create model 
model = Sequential([
    Linear(n_features=2, n_neurons=3), 
    Activation('tanh'), 
    Linear(n_features=3, n_neurons=1), 
    Activation('sigmoid') 
])

# Define loss and optimizer
loss = Loss('mse')
optimizer = SGD(model, learning_rate=0.01, momentum= 0.9)

verbose = True    
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model.forward(X)

    # Loss calculation
    error = loss.forward(y, y_pred)

    # Compute gradient
    grad = loss.backward(y, y_pred)

    # Backward pass
    model.backward(grad)

    # Update parameters
    optimizer.step()

    # Check and print gradients for each layer
    if verbose and (epoch % 100) == 99: 
        print(f'Epoch: {epoch + 1}, Error = {error}')


