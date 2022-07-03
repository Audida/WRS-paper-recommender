import math
import tensorflow as tf
from tensorflow import keras
import numpy as np

 
class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, av_dict, uv_dict, batch_size=1024, dim=300, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.X = X # dataset userID articleID
        self.shuffle=shuffle
        self.av_dict = av_dict
        self.uv_dict = uv_dict
        self.indexes = X.index
        self.on_epoch_end()
 
    def __len__(self): 
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
 
    def __get_vectors(self, df):   
        x_uv=np.zeros(shape=(self.batch_size, self.dim)) 
        x_av=np.zeros(shape=(self.batch_size, self.dim))  
        y_l=np.zeros(shape=self.batch_size)  
        
        for i in range(0,len(df)):
            x_uv[i] = self.uv_dict[df[i][0]]
            x_av[i] = self.av_dict[df[i][1]]
            y_l[i] = df[i][2]     
          
        return x_uv,x_av, y_l


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = [i for i in range(index*self.batch_size,(index+1)*self.batch_size)]

        #workaround
        idxs = [x for x in idxs if x < len(self.X)]
        
        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in idxs]
        
        # Generate data
        dataset=self.X.reindex(list_IDs_temp)
        x1, x2, y=self.__get_vectors(dataset.to_numpy()) 
            
            
        return [x1, x2], y
 
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
