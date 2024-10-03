import numpy as np 
class Module():
    def __init__(self): 
        self.grads = dict() 


    #TODO: take forward for neuron network 
    def forward(): 
        pass 

    #TODO: compute gradint 
    def backward():  
        pass


class Sequential(): 
    def __init__(self, layers): 
        self.layers = layers 

    def forward(self,x):
        self.x  = x      # Save input array 
        for layer in self.layers: 
           x = layer.forward(x) 

        return x 

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout

    def __iter__(self):
        return iter(self.layers)  