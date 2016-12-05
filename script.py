'''
Created on Dec 1, 2016

@author: eao
'''
import numpy as np
#import matplotlib.pyplot as plt
from NearestNeighbor import *

print("Hello World:)\n")

ds=DataSet()
Itr, Ltr, Ite, Lte = ds.load_CIFAR10() 
#Xtr, Ytr, Xte, Yte = ds.load_CIFAR10() 
#image=ARR['data']
#Itr_rows = Itr.reshape(Itr.shape[0], 32 * 32 * 3)
#Ite_rows = Ite.reshape(Ite.shape[0], 32 * 32 * 3)


print (Itr.shape, Ltr.shape, Ite.shape)
# 
# Itr=Itr[:100,:]
# Ltr=Ltr[:100]
# 
# print (Itr.shape, Ltr.shape, Ite.shape)

#image=ARR['data'][5]
# im = numpy.reshape(image, (32,32,3), order='F')
#im = image.reshape(3,32,32).transpose(1,2,0)
#plt.imshow(im)

nn = NearestNeighborClass() # create a Nearest Neighbor classifier class
nn.train(Itr, Ltr) # train the classifier on the training images and labels
Lte_predict = nn.predictL1(Ite) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Lte_predict == Lte) )

