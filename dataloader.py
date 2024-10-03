import numpy as np 
class DataLoader: 
    def __init__(self, X, y, batchsize, shuffle= True): 
        """
        Dataloader to split data into batch 
        Parameters: 
            - X (np.array): features 
            - y (np.array): labels 
            - batchsize (int): the batch_size want to split 
            - shuffle (bool): should shuffe data or not  
          
        """
        combined_data = list(zip(X,y))

        if shuffle: 
            np.random.shuffle(combined_data)

        batches = [ combined_data[i : i + batchsize] for i in range(0, len(combined_data), batchsize)]
        self.batch_data = [ ([item[0] for item in batch], [item[1] for item in batch]) for batch in batches] 

    def __iter__(self): 
        """
        Make Dataloader itereable 
        """
        return iter(self.batch_data)
    
    def __len__(self):
        """
        Return the number of batches
        """
        return len(self.batch_data)