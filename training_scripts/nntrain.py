#!/usr/bin/python

'''
nntrain.py - Training script for the precached sample.

RPI Rock Raiders
5/18/15

Last Updated: Bryant Pong: 5/23/15 - 1:54 PM
'''

# Python Imports:
import cPickle as pickle # cPickle is faster for Python 2.x  
import theano # Python symbolics library
import theano.tensor as T # By convention theano's tensor is T  
import numpy as np
import lasagne # Convolutional Neural Network Library
import os # For open() function
import warnings # For ignoring any lasagne warnings
import cv2 # For resizing images
import matplotlib.pyplot as plt

'''
This function dumps the training data of images into chunks.  It resizes the images
(which are initially  
'''
def dumpData():
	
	# The training images and labels:
	imgs, labels = [], []	
	folderName = "/home/bryant/rockie_neural_network/data/pickle/"

	# A list of pickle files to open:
	fileNames = ["samplelight1.dat", "samplelight2.dat", "negatives1.dat", "negatives2.dat"] 

	for fileName in fileNames:

		with open(folderName+fileName, "rb") as f:

			print("Now processing: " + str(fileName))

			data = pickle.load(f)

			for img in data:
				imgs.append(cv2.resize(img[0], (128, 72)))
				if len(img[1].tolist()) == 0:
					labels.append(0)
				else:
					labels.append(1)

			print("Done processing: " + str(fileName))

	# Divide the resized data into 2 sections:
	resizedImgs1 = pickle.dump(imgs[:len(imgs)/2], open("resizedImgs1.dat","wb"))
	resizedImgs2 = pickle.dump(imgs[len(imgs)/2:], open("resizedImgs2.dat","wb"))

	resizedLabels1 = pickle.dump(labels[:len(labels)/2], open("resizedLabels1.dat","wb"))
	resizedLabels2 = pickle.dump(labels[len(labels)/2:], open("resizedLabels2.dat","wb"))

'''
This function loads in the training data for training the neural network.   
'''
def loadData():
	'''
	The files to load.  Data is in the form of a list of numpy matrices.  
	'''
	folderName = "/home/bryant/rockie_neural_network/data/pickle/"
	images = ["resizedImgs1.dat", "resizedImgs2.dat"]
	labels = ["resizedLabels1.dat", "resizedLabels2.dat"]
	
	# Piece together the images and labels:
	allImgs, allLabels = [], []	  

	for imageset in images:
		with open(folderName+imageset, "rb") as f:
			nextData = pickle.load(f) 	
			allImgs.extend(nextData)

	for labelset in labels:
		with open(folderName+labelset, "rb") as f:
			nextData = pickle.load(f)
			allLabels.extend(nextData)

	return np.array(allImgs), np.array(allLabels).astype(np.int32)

# Training function:
def train():
	
	# Deactivate warnings for lasagne:
	warnings.filterwarnings("ignore", module="lasagne")

	# Load the training data of the precached sample: 
	print("Now loading training data")
	X, y = loadData()
	print("X shape: " + str(X.shape))
	print("y shape: " + str(y.shape))
	# Reshape the dataset:
	X = X.reshape((X.shape[0], 3, 360, 240))

	plt.imshow(X[0])
	plt.show()

	cv2.waitKey(0)
	
	print("Complete loading training data")

	'''
	Create the Neural Network.  There will be 6 layers in this network:
	1) Input layer - Data takes the form of (None, 3 (for RGB), 360, 240)
	2) Convolutional Layer   	 
	3) Pooling Layer
	4) Convolutional Layer
	5) Pooling Layer 
	'''
	# Input layer:
	l_in = lasagne.layers.InputLayer(
		shape=(None,3,360,240))	

	'''
	First convolutional layer:
	Parameters:
	filter_size = 3x3
	num_filters = 8 
	'''
	l_conv1 = lasagne.layers.Conv2DLayer(
		l_in,
		num_filters=8,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))

	'''
	First pooling layer. 
	Parameters:
	filter_size = 2x2 
	'''
	l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))

	'''
	Second convolutional layer:
	Parameters:
	filter_size = 3x3
	num_filters = 16
	'''
	l_conv2 = lasagne.layers.Conv2DLayer(
		l_pool1, 
		num_filters=16, 
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))

	'''
	Second pooling layer:
	Parameters:
	filter_size = 2x2	 
	'''
	l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2))

	'''
	Output layers:
	'''
	l_hidden1 = lasagne.layers.DenseLayer(
		l_pool2, num_units=256,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))

	l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
	l_output = lasagne.layers.DenseLayer(
		l_hidden1_dropout,
		num_units=10,
		nonlinearity=lasagne.nonlinearities.softmax)

	true_output = T.ivector('true_output')
	objective = lasagne.objectives.Objective(l_output,
		loss_function=lasagne.objectives.categorical_crossentropy)

	loss_train = objective.get_loss(target=true_output, deterministic=False)
	loss_eval = objective.get_loss(target=true_output, deterministic=True)

	all_params = lasagne.layers.get_all_params(l_output)
	updates = lasagne.updates.adadelta(loss_train, all_params)
	train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)

	get_output = theano.function([l_in.input_var], l_output.get_output(deterministic=True)) 

	print("Now beginning to train convolutional neural network")
	train(X, y)
	print("Done training neural network")

	# Validate the training set:
	for image in X:
		predict = get_output(image)

		print("Prediction is: " + str(predict))

# Main function runner:
if __name__ == "__main__":

	# Get the command line arguments:
	train()
