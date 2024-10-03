# Utils.py 
import numpy as np 

class DataLoader:
    def __init__(self, data, labels, batchsize, shuffle=True):
        """
        Dataloader to split data into batches.
        
        Parameters:
            - data (np.array): Features array (X)
            - labels (np.array): Labels array (y)
            - batchsize (int): The batch size for splitting the data
            - shuffle (bool): Whether to shuffle data before batching
        """
        # Combine data and labels
        combined_data = list(zip(data, labels))

        # Shuffle the data if specified
        if shuffle:
            np.random.shuffle(combined_data)

        # Create batches of data
        batches = [combined_data[i: i + batchsize] for i in range(0, len(combined_data), batchsize)]
        
        # Split each batch into feature and label arrays
        self.batch_data = [
            (np.array([item[0] for item in batch]), np.array([item[1] for item in batch])) 
            for batch in batches
        ]

    def __iter__(self):
        """
        Make DataLoader iterable.
        """
        return iter(self.batch_data)

    def __len__(self):
        """
        Return the number of batches.
        """
        return len(self.batch_data)
    

class StandardScaler(): 
    def __init__(self, ):
        self.mean_x = None 
        self.std_x = None 

    def fit(self, X): 
        self.mean_x = np.mean(X, axis= 0, keepdims= True) 
        self.std_x = np.std(X, axis= 0, keepdims= True)  + 1e-12 # Ensure not devision byy zero 

    def transform(self, X): 
        X_norm = ( X - self.mean_x )  / self.std_x 
        return X_norm

    def inverse_transform(self, X_norm): 
        X = X_norm * self.std_x + self.mean_x 
        return X 
    
    def fit_transform(self, X): 
        self.fit(X) 
        return self.transform(X)

class OnehotEncoder():
    def __init__(self):
        self.unique_classes = None 
        self.labels = None 
        self.encoded_labels = None 

    def fit(self, labels):
        self.labels = labels # Save labels 

        # Unique class in this labels 
        self.unique_classes = np.unique(self.labels) 
        print(f'Unique classes: {self.unique_classes}')

    def transform(self): 
        # Convert unique_class to index (int number) 
        class_to_index = {cls: idx for idx, cls in enumerate(self.unique_classes)}

        # Create indices array ( map class -> int )
        indices = [class_to_index[label] for label in self.labels]

        # Encoded matrix 
        self.encoded_labels = np.zeros(shape= (len(self.labels), len(self.unique_classes)))
        self.encoded_labels[np.arange(len(self.labels)), indices] = 1 

        return self.encoded_labels 
    
    def inverse_transform(self,y_onehot ): 
        # Convert idx -> class 
        index_to_class = {idx : cls for idx, cls in enumerate(self.unique_classes) }

        # Create indices array ( map idx -> class ) 
        indices = np.argmax(y_onehot, axis= 1) 
        
        # Original matrix 
        original_labels = [index_to_class[idx] for idx in indices ] 

        return np.array(original_labels)
