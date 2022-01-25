import numpy as np 
import math 
from math import * 

class KNN_classifier:
    def __init__ (self):
        pass

    def train(self, X,y):
        # creating a training set 
        self.X_train = X
        self.y_train = y 

    def predict(self, X, k = 1, nums_loop = 0):

        if nums_loop == 0:
            dist = self.compute_distance_no_loop(X)
        elif nums_loop == 1:
            dist = self.compute_distance_one_loop(X)
        elif nums_loop == 2:
            dist = self.compute_distance_two_loop(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % nums_loop)   

        return self.pred_labels(dist,k=k) 

    
    def compute_distance_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros(num_test, num_train)
        for i in num_test:
            for j in num_train:
                dist[i,j] = np.sqrt(np.sum(np.abs(X[i,:] - self.X_train[j,:])))
                pass

        return dist

    def compute_distance_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros(num_test, num_train)
        
        for i in num_test:
            dist[i,:] = np.sqrt(np.sum(np.abs(X[i:,] - self.X_train), axis = 1))
            pass 
        return dist

    
    def compute_distance_no_loop(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros(num_test, num_train)
        

        test_sum = np.sum(np.square(X), axis = 1)
        train_sum = np.sum(np.square(self.X_train), axis =1)
        inner_prod = np.dot(X, self.X_train.T)
        dist = np.sqrt(-2*inner_prod +train_sum + test_sum.reshape(-1,1))

        pass 
        return dist

    def pred_labels(self, dist, k =1):
        num_test = dist.shape[0]
        y_pred = np.zeros(num_test)

        for i in num_test:
            y_indices = np.argsort(dist[i,:], axis = 0)
            closet_y = self.y_train[y_indices[:k]]

            y_pred[i] = np.argmax(np.bincount(closet_y))
        
        return y_pred

