import numpy as np
import matplotlib.pyplot as plt

class Spiral:
    def __init__(self, n_points, n_classes, n_dimensions):        
        #generating a randomized data
        self.N = n_points # number of points per class
        self.D = n_dimensions # dimension
        self.K = n_classes # number of classes
        self.P = np.zeros((self.N*self.K,self.D)) # data matrix (each row = single example)
        self.L = np.zeros(self.N*self.K, dtype='uint8') # class labels
        for j in range(self.K):
            ix = range(self.N*j,self.N*(j+1))
            r = np.linspace(0.0,1,self.N) # radius
            t = np.linspace(j*4,(j+1)*4,self.N)  + np.random.randn(self.N)*0.2 # theta
            self.P[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.L[ix] = j

    def generate(self):
        pass
        
###################################################################      

class Line:
    def __init__(self, n_points, n_classes, n_dimensions):        
        #generating a randomized data
        self.N = n_points # number of points per class
        self.D = n_dimensions # dimension
        self.K = n_classes # number of classes
        self.P = np.zeros((self.N*self.K,self.D)) # data matrix (each row = single example)
        self.L = np.zeros(self.N*self.K, dtype='uint8') # class labels
        for j in range(self.K):
            a = 2*(j-1) 
            b = np.zeros(self.N)
            for i in range(len(b)):
                b[i] = (j-2)*2 + np.random.randn(1)
            ix = range(self.N*j,self.N*(j+1))
            t = np.linspace(-10,10,self.N)
            if (self.D == 2):
                self.P[ix] = np.c_[t, a*t + b]
                self.L[ix] = j

    def generate(self):
        pass
        

###################################################################      

class Circle:
    def __init__(self, n_points, n_classes, n_dimensions):        
        #generating a randomized data
        self.N = n_points # number of points per class
        self.D = n_dimensions # dimension
        self.K = n_classes # number of classes
        self.P = np.zeros((self.N*self.K,self.D)) # data matrix (each row = single example)
        self.L = np.zeros(self.N*self.K, dtype='uint8') # class labels
        for j in range(self.K):
            ix = range(self.N*j,self.N*(j+1))
            r = np.zeros(self.N)
            for i in range(len(r)):
                r[i] = (j+1)*2 + np.random.randn(1)*0.5
            t = np.linspace(0,2*3.1415,self.N)  + np.random.randn(self.N)*0.2 # theta
            self.P[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.L[ix] = j

    def generate(self):
        pass

###################################################################      

class Zone:
    def __init__(self, n_points, n_classes, n_dimensions):        
        #generating a randomized data
        self.N = n_points # number of points per class
        self.D = n_dimensions # dimension
        self.K = n_classes # number of classes
        self.P = np.zeros((self.N*self.K,self.D)) # data matrix (each row = single example)
        self.L = np.zeros(self.N*self.K, dtype='uint8') # class labels
        pi = 3.1415
        for j in range(self.K):
            theta = j*(2*pi)/self.K
            a = np.cos(theta)
            b = np.sin(theta)
            ix = range(self.N*j,self.N*(j+1))
            r = np.zeros(self.N)
            for i in range(len(r)):
                r[i] = np.random.randn(1)*0.5
            t = np.linspace(0,2*3.1415,self.N)  + np.random.randn(self.N)*0.2 # theta
            self.P[ix] = np.c_[a + r*np.sin(t), b + r*np.cos(t)]
            self.L[ix] = j

    def generate(self):
        pass

###################################################################    

class Zone_3D:
    def __init__(self, n_points, n_classes, n_dimensions, centers):        
        #generating a randomized data
        self.N = n_points # number of points per class
        self.D = n_dimensions # dimension
        self.K = n_classes # number of classes
        self.P = np.zeros((self.N*self.K,self.D)) # data matrix (each row = single example)
        self.L = np.zeros(self.N*self.K, dtype='uint8') # class labels
        pi = 3.1415
        for j in range(self.K):
            center_j = centers[j]
            theta = j*(2*pi)/self.K
            a = np.cos(theta)
            b = np.sin(theta)
            ix = range(self.N*j,self.N*(j+1))
            R = np.zeros(self.N)
            k = np.zeros(self.N)
            r = np.zeros(self.N)
            g = np.linspace(0,2*3.1415,self.N)  #+ np.random.randn(self.N)*0.2 # gamma
            t = np.linspace(0,2*3.1415,self.N)  #+ np.random.randn(self.N)*0.2 # theta

            for i in range(len(R)):
                R[i] = np.random.randn(1)*2
                k[i] = R[i] * np.sin(g[i])
                r[i] = R[i] * np.cos(g[i])

            self.P[ix] = np.c_[center_j[0] + r*np.sin(t), center_j[1] + r*np.cos(t), center_j[2] + k]
            self.L[ix] = j

    def generate(self):
        pass