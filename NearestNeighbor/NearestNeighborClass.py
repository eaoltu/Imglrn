'''
Created on Dec 1, 2016

@author: eao
'''

import numpy as np
import progressbar
#from time import sleep

class NearestNeighborClass(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predictL1(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        bar = progressbar.ProgressBar(maxval=num_test, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            # using the L2 distance (sum of absolute value differences)
            #distances = np.sum(np.square(self.Xtr - X[i,:]), axis = 1)
            
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            bar.update(i+1)
        bar.finish()
        return Ypred
    
    def predictL2(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        bar = progressbar.ProgressBar(maxval=num_test, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L2 distance (sum of absolute value differences)
            distances = np.sum(np.square(self.Xtr - X[i,:]), axis = 1)
            
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            bar.update(i+1)
        bar.finish()

        return Ypred


        def predictK(self, X, k):
            """ X is N x D where each row is an example we wish to predict label for """
            num_test = X.shape[0]
            # lets make sure that the output type matches the input type
            Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

            # loop over all test rows
            for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            # using the L2 distance (sum of absolute value differences)
            #   distances = np.sum(np.square(self.Xtr - X[i,:]), axis = 1)
            
                min_index = np.argpartition(distances, k) # get the index with smallest distance
                Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

            return Ypred

