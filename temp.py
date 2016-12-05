#Xtr, Ytr, Xte, Yte = ds.load_CIFAR10() 
#image=ARR['data']
Itr_rows = Itr.reshape(Itr.shape[0], 32 * 32 * 3)
Ite_rows = Ite.reshape(Ite.shape[0], 32 * 32 * 3)
print (Itr.shape, Ltr.shape)
#image=ARR['data'][5]
# im = numpy.reshape(image, (32,32,3), order='F')
#im = image.reshape(3,32,32).transpose(1,2,0)
#plt.imshow(im)

nn = NearestNeighborClass() # create a Nearest Neighbor classifier class
nn.train(Itr_rows, Ltr) # train the classifier on the training images and labels
Lte_predict = nn.predict(Ite_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Lte_predict == Lte) )

