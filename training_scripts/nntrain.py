#!/usr/bin/python

'''
nntrain.py - Training script for a neural network to recognize the precached
sample.

RPI Rock Raiders
5/27/15

Last Updated: Bryant Pong: 5/28/15 - 9:24 PM
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
imgNames = ["negativesImgs1.dat", "negativesImgs2.dat", "sample_lightImgs1.dat", "sample_lightImgs2.dat", "sample_shadowImgs1.dat", "sample_shadowImgs2.dat"]
labelNames = ["negativesLabels1.dat", "negativesLabels2.dat", "sample_lightLabels1.dat", "sample_lightLabels2.dat", "sample_shadowLabels1.dat", "sample_shadowLabels2.dat"]
validImgs = ["negativesvalid.dat", "sample_lightvalid.dat", "sample_shadowvalid.dat"]
validLabels = ["negativesvalidLabels.dat", "sample_lightvalidLabels.dat", "sample_shadowvalidLabels.dat"]
imgWidth = 224
imgHeight = 224

def loadData():
	X, y = [], []

	for imgSet in imgNames:
		with open(path+imgSet, "rb") as f:
			for i in pickle.load(f):
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
				if len(l) > 0:
					y.append(1)
				else:
					y.append(0)

	return np.array(X), np.array(y)

def train():

	# Suppress warnings from lasagne
	warnings.filterwarnings('ignore', module='lasagne')

	# Load the training data:
	print("Now loading training data")
	X, y = loadData() 		
	print("Done loading training data")

	print("Now loading validation data")
	validX, validY = loadValidationData()
	print("Done loading validation data")

	print("X.shape: " + str(X.shape))
	print("y.shape: " + str(y.shape))

	X = X.astype(theano.config.floatX)
	y = y.astype(np.int32)

	'''
	print("Verify validation images")
	for i in xrange(len(validX)):
		print(str(validY[i]))
		plt.imshow(np.reshape(validX[i], (imgHeight, imgWidth, 3)))
		plt.show()
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
		num_filters=96,
		filter_size=(11,11),
		stride=4,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))
	l_conv2 = lasagne.layers.Conv2DLayer(
		l_pool1,
		num_filters=256,
		filter_size=(5,5),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))	
	l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2))
	l_conv3 = lasagne.layers.Conv2DLayer(
		l_pool2,
		num_filters=384,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_conv4 = lasagne.layers.Conv2DLayer(
		l_conv3,
		num_filters=384,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_conv5 = lasagne.layers.Conv2DLayer(
		l_conv4, 
		num_filters=256,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv5, pool_size=(2,2))
	l_hidden1 = lasagne.layers.DenseLayer(
		l_pool3,
		num_units=256,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_hidden2 = lasagne.layers.DenseLayer(
		l_hidden1,
		num_units=256,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_output = lasagne.layers.DenseLayer(
		l_hidden2,
		num_units=2,
		nonlinearity=lasagne.nonlinearities.softmax)
	'''
	l_hidden1 = lasagne.layers.DenseLayer(
		l_pool6, 
		num_units=512,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))		 
	l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
	l_output = lasagne.layers.DenseLayer(
		l_hidden1_dropout,
		num_units=2,
		nonlinearity=lasagne.nonlinearities.softmax)
	'''
	true_output = T.ivector('true_output')
	objective = lasagne.objectives.Objective(l_output,
			loss_function=lasagne.objectives.categorical_crossentropy)
	loss_train = objective.get_loss(target=true_output, deterministic=False)
	loss_eval = objective.get_loss(target=true_output, deterministic=True)

	all_params = lasagne.layers.get_all_params(l_output)
	updates = lasagne.updates.adadelta(loss_train, all_params)
	train = theano.function([l_in.input_var, true_output], loss_train, updates=updates)
	
	get_output = theano.function([l_in.input_var], l_output.get_output(deterministic=True))

	print("Now training neural net")
	blockIdx = 100
	epoch = 1
	epochs = 100
	batchIdx = 0
	output = []
	highestAccuracy = 0.0
	while epoch < epochs:
		train(X[batchIdx:batchIdx+blockIdx], y[batchIdx:batchIdx+blockIdx])
		batchIdx += blockIdx

		print("Epoch: " + str(epoch))
		print("batchIdx: " + str(batchIdx))
		if batchIdx >= X.shape[0]:

			#results = get_output([X]) 		

			print("Now validating neural net")
			numCorrect = 0 
			for i in xrange(validX.shape[0]):
				predict = get_output([validX[i]]) 
				#print("Predict: " + str(predict))
				maxArg = np.argmax(predict[0])
				#print("Max argument: " + str(maxArg))
				#print("y[i]: " + str(y[i]))
				
				if maxArg == validY[i]:
					numCorrect += 1
		
			print("Percent correct: " + str(float(numCorrect) / validX.shape[0]) + " for Epoch: " + str(epoch)) 
			print("Done validating neural net")			
			
			epoch += 1
			batchIdx = 0

			if float(numCorrect) / validX.shape[0] > highestAccuracy:
				print("This Epoch has the highest accuracy!")
				highestAccuracy = float(numCorrect) / validX.shape[0]
				output = lasagne.layers.get_all_params(l_output)

	pickle.dump(output, open("output.dat", "wb"))
	pickle.dump(highestAccuracy, open("highestAccuracy", "wb"))
	print("Highest accuracy produced: " + str(highestAccuracy))
	print("All done")

if __name__ == "__main__":
	train()
