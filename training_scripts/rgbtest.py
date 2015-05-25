#!/usr/bin/python

import lasagne
from lasagne import layers
from lasagne import nonlinearities

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 

import theano
import theano.tensor as T

import warnings

path = "/home/bryant/rockie_neural_network/data/images/"
imgs = [cv2.imread(path+img) for img in sorted(os.listdir(path))]

#print("imgs: " + str(imgs))

def main():

	warnings.filterwarnings('ignore', module='lasagne')

	imgWidth = 360
	imgHeight = 240 	
	
	X = np.zeros((200, 3, imgHeight, imgWidth))
	labels = []
	labels.extend([1 for i in xrange(42)])
	labels.extend([0,0])
	labels.extend([1,1,1])
	labels.extend([0, 0, 0, 0, 0])
	labels.extend([1, 1, 1, 1])
	labels.extend([0, 0])
	labels.extend([1 for i in xrange(42)])
	labels.extend([0 for i in xrange(100)])
	y = np.array(labels)
	
	# Open the images:
	for i in xrange(200):
		img = imgs[i]
		img = img.reshape(3, imgHeight, imgWidth)
		X[i] = img


	X = X.astype(theano.config.floatX)
	y = y.astype(np.int32)

	print("X.shape: " + str(X.shape))
	print("y.shape: " + str(y.shape))

	print("Now creating Neural Network")
	l_in = lasagne.layers.InputLayer(
		shape=(None, 3, imgHeight, imgWidth))
	l_conv1 = lasagne.layers.Conv2DLayer(
		l_in,
		num_filters=16,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2,2))
	l_conv2 = lasagne.layers.Conv2DLayer(
		l_pool1, 
		num_filters=16,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2,2))
	l_hidden1 = lasagne.layers.DenseLayer(
		l_pool2, 
		num_units=128,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))		 
	l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
	l_output = lasagne.layers.DenseLayer(
		l_hidden1_dropout,
		num_units=2,
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

	print("Now training neural net")
	blockIdx = 5
	epoch = 1
	epochs = 101
	batchIdx = 0
	while epoch < epochs:
		print("Epoch: " + str(epoch))
		print("batchIdx: " + str(batchIdx))
		train(X[batchIdx:batchIdx+blockIdx], y[batchIdx:batchIdx+blockIdx])
		batchIdx += blockIdx

		if batchIdx >= 200:

			#results = get_output([X]) 		

			print("Now validating neural net")
			numCorrect = 0 
			for i in xrange(200):
				predict = get_output([X[i]]) 
				#print("Predict: " + str(predict))
				maxArg = np.argmax(predict[0])
				#print("Max argument: " + str(maxArg))
				#print("y[i]: " + str(y[i]))
				
				if maxArg == y[i]:
					numCorrect += 1
		
			print("Percent correct: " + str(float(numCorrect) / 200) + " for Epoch: " + str(epoch)) 
			print("Done validating neural net")			
			
			epoch += 1
			batchIdx = 0
	print("All done")

if __name__ == "__main__":
	main()
