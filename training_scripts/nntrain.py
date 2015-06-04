#!/usr/bin/python

'''
nntrain.py - Training script for a neural network to recognize the precached
sample.

RPI Rock Raiders
5/27/15

Last Updated: Bryant Pong: 6/4/15 - 12:01 PM
'''

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import theano
import theano.tensor as T
import cPickle as pickle
import warnings

path = "../data/pickle/"
imgNames = ["negativesImgs1.dat", "sample_lightImgs1.dat", "sample_shadowImgs1.dat", \
"negativesImgs1_flipped.dat", "sample_lightImgs1_flipped.dat", "sample_shadowImgs1_flipped.dat", \
"negativesImgs2.dat", "sample_lightImgs2.dat", "sample_shadowImgs2.dat", \
"negativesImgs2_flipped.dat", "sample_lightImgs2_flipped.dat", "sample_shadowImgs2_flipped.dat"]
labelNames = ["negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels.dat", \
"negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat"]
validImgs = ["negativesvalid.dat", "sample_lightvalid.dat", "sample_shadowvalid.dat"]
validLabels = ["negativesvalidLabels.dat", "sample_lightvalidLabels.dat", "sample_shadowvalidLabels.dat"]
imgWidth = 224
imgHeight = 224

def loadData():
	X, y = [], []

	for imgSet in imgNames:
		with open(path+imgSet, "rb") as f:
			for i in pickle.load(f):
				#plt.imshow(i)
				#plt.show()
				resizedImg = i.reshape(3, imgHeight, imgWidth) 
				X.append(resizedImg)
	for label in labelNames:
		with open(path+label, "rb") as f:
			for l in pickle.load(f):
				if len(l) > 0:
					y.append(1)
				else:
					y.append(0) 	  			
			
	return np.array(X), np.array(y)

def loadValidationData():
	X, y = [], []
	
	for imgSet in validImgs:
		with open(path+imgSet, "rb") as f:
			for i in pickle.load(f):
				resizedImg = i.reshape(3, imgHeight, imgWidth)
				X.append(resizedImg)
	for label in validLabels:
		with open(path+label, "rb") as f:
			for l in pickle.load(f):

				print("l: " + str(l))
				if len(l) > 0:
					print("Appending 1")
					y.append(1)
				else:
					print("Appending 0")
					y.append(0)

	return np.array(X), np.array(y)

'''
This function calculates the training loss for a given Epoch.
'''
def calcTrainingLoss(epochParams, batchSize, totalSamples):

	numBatches = int(totalSamples / batchSize)

	#for b in xrange(

	

def train():

	# Suppress warnings from lasagne
	warnings.filterwarnings('ignore', module='lasagne')

	# Load the training data:
	'''
	print("Now loading training data")
	X, y = loadData() 		
	print("Done loading training data")
	'''
	print("Now loading validation data")
	validX, validY = loadValidationData()
	validX = validX.astype(theano.config.floatX)
	validY = validY.astype(np.int32)
	print("Done loading validation data")

	'''
	print("X.shape: " + str(X.shape))
	print("y.shape: " + str(y.shape))

	X = X.astype(theano.config.floatX)
	y = y.astype(np.int32)
	'''

	'''
	Neural Network Architecture (Borrowed from ImageNet Paper):
	1) Input Layer (Accepting list of images of 3 x 184 x 184)
	2) Convolutional Layer 1 (96 filters of size 11x11 / stride 4)    
	3) Pooling Layer 1 (2 x 2 filter)
	4) Convolutional Layer 2 (256 filters of size 5x5) 
	'''
	print("Now creating Neural Network")
	l_in = lasagne.layers.InputLayer(
		shape=(None, 3, imgHeight, imgWidth))
	l_conv1 = lasagne.layers.Conv2DLayer(
		l_in,
		num_filters=12,
		filter_size=(11,11),
		stride=4,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_pool1 = lasagne.layers.MaxPool2DLayer(
		l_conv1, 
		pool_size=(3,3),
		stride=2)
	l_conv2 = lasagne.layers.Conv2DLayer(
		l_pool1,
		num_filters=32,
		filter_size=(5,5),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))	
	l_pool2 = lasagne.layers.MaxPool2DLayer(
		l_conv2, 
		pool_size=(3,3),
		stride=2)
	l_conv3 = lasagne.layers.Conv2DLayer(
		l_pool2,
		num_filters=48,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_conv4 = lasagne.layers.Conv2DLayer(
		l_conv3,
		num_filters=48,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_conv5 = lasagne.layers.Conv2DLayer(
		l_conv4, 
		num_filters=32,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_pool3 = lasagne.layers.MaxPool2DLayer(
		l_conv5, 
		pool_size=(3,3),
		stride=2)
	l_hidden1 = lasagne.layers.DenseLayer(
		l_pool3,
		num_units=128,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_dropout1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
	l_hidden2 = lasagne.layers.DenseLayer(
		l_dropout1,
		num_units=128,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_dropout2 = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)
	l_output = lasagne.layers.DenseLayer(
		l_dropout2,
		num_units=1,
		nonlinearity=lasagne.nonlinearities.sigmoid)
	true_output = T.iscalar('true_output')
	objective = lasagne.objectives.Objective(l_output,
			loss_function=lasagne.objectives.binary_crossentropy)
	loss_train = objective.get_loss(target=true_output, deterministic=False)
	loss_eval = objective.get_loss(target=true_output, deterministic=True)

	all_params = lasagne.layers.get_all_params(l_output)
	#updates = lasagne.updates.adadelta(loss_train, all_params)
	updates = lasagne.updates.nesterov_momentum(
		loss_train, all_params, 0.01, 0.9)
	train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)
	
	get_output = theano.function([l_in.input_var], l_output.get_output(deterministic=True))

	# Display the number of parameters that this neural network has:
	final_params = lasagne.layers.get_all_param_values(l_output)

	'''
	numParams = 0
	print("params: " + str(final_params))
	for p in final_params:
		print("type of p: " + str(type(p)))
		print(p)
		numParams += len(p.flatten())

	print("number of parameters: " + str(numParams))
	'''

	# Variables for the neural network:
	blockIdx = 1
	epoch = 1
	epochs = 4
	xAxis, yAxis = [], []
	batchIdx = 0
	output = []
	highestAccuracy = 0.0

	print("Now beginning training of neural network")	

	for i in xrange(len(imgNames)):
		# Get the next image set and label set:
		print("Now extracting data on image set: " + str(imgNames[i]) + " and label set: " + str(labelNames[i]))
		X, y = [], []
		with open(path+imgNames[i], "rb") as imgX, open(path+labelNames[i], "rb") as lblY:
			for nextImg in pickle.load(imgX):
				# Reshape the image:
				X.append(nextImg.reshape(3,imgHeight,imgWidth))
				print("X.shape: " + str(nextImg.reshape(3,imgHeight,imgWidth).shape))
			for nextLbl in pickle.load(lblY):
				if len(nextLbl) > 0:
					y.append(1)
				else:
					y.append(0)

		X = np.array(X[:10])
		y = np.array(y[:10])

		# Cast X and y:
		X = X.astype(theano.config.floatX)
		y = y.astype(np.int32)

		print("Now training on image set: " + str(imgNames[i]))  
		print("Size of set: " + str(len(X)))

		epoch = 1
		while epoch < epochs:

			train(X[batchIdx:batchIdx+blockIdx], y[batchIdx:batchIdx+blockIdx])
			batchIdx += blockIdx

			print("Epoch: " + str(epoch) + " " + str(imgNames[i]))
			print("batchIdx: " + str(batchIdx))
			if batchIdx >= X.shape[0]:

				#results = get_output([X]) 		

				print("Now validating neural net")
				numCorrect = 0 

				#predict = get_output(validX)
				#print("Global predict: " + str(predict))

				for validItr in xrange(validX.shape[0]):
					predict = get_output([validX[validItr]]) 
					print("Predict: " + str(validItr) + " " + str(predict))
					maxArg = np.max(predict)
					print("Max argument: " + str(maxArg))
					#plt.imshow(validX[validItr].reshape(imgHeight, imgWidth, 3))
					#plt.show()
					print("y[i]: " + str(validY[validItr]))
				
					if maxArg == validY[validItr]:
						numCorrect += 1
		
				print("Percent correct: " + str(float(numCorrect) / validX.shape[0]) + " for Epoch: " + str(epoch)) 
				print("Done validating neural net")			
			
				epoch += 1
				batchIdx = 0

				if float(numCorrect) / validX.shape[0] > highestAccuracy:
					print("This Epoch has the highest accuracy!")
					highestAccuracy = float(numCorrect) / validX.shape[0]
					output = lasagne.layers.get_all_param_values(l_output)

					pickle.dump(output, open("output.dat", "wb"))
					pickle.dump(highestAccuracy, open("highestAccuracy", "wb"))

				#xAxis.append(epoch)
				#yAxis.append(float(numCorrect) / validX.shape[0])	

				# Plot the validation loss:
				#plt.plot(xAxis, yAxis)
				#plt.show()	 
			
	print("All done")

if __name__ == "__main__":
	train()
