#!/usr/bin/python

'''
nntrain.py - Training script for a neural network to recognize the precached
sample.

RPI Rock Raiders
5/27/15

Last Updated: Bryant Pong: 6/4/15 - 2:22 PM
'''

import lasagne
import os
import itertools
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import theano
import theano.tensor as T
import cPickle as pickle
import warnings
import time

path = "../data/pickle/"
imgNames = ["negativesImgs1.dat", "sample_lightImgs1.dat", "sample_shadowImgs1.dat", \
"negativesImgs1_flipped.dat", "sample_lightImgs1_flipped.dat", "sample_shadowImgs1_flipped.dat", \
"negativesImgs2.dat", "sample_lightImgs2.dat", "sample_shadowImgs2.dat", \
"negativesImgs2_flipped.dat", "sample_lightImgs2_flipped.dat", "sample_shadowImgs2_flipped.dat", \
"negativesImgs1.dat", "sample_lightImgs1.dat", "sample_shadowImgs1.dat", \
"negativesImgs1_flipped.dat", "sample_lightImgs1_flipped.dat", "sample_shadowImgs1_flipped.dat", \
"negativesImgs2.dat", "sample_lightImgs2.dat", "sample_shadowImgs2.dat", \
"negativesImgs2_flipped.dat", "sample_lightImgs2_flipped.dat", "sample_shadowImgs2_flipped.dat", \
"negativesImgs1.dat", "sample_lightImgs1.dat", "sample_shadowImgs1.dat", \
"negativesImgs1_flipped.dat", "sample_lightImgs1_flipped.dat", "sample_shadowImgs1_flipped.dat", \
"negativesImgs2.dat", "sample_lightImgs2.dat", "sample_shadowImgs2.dat", \
"negativesImgs2_flipped.dat", "sample_lightImgs2_flipped.dat", "sample_shadowImgs2_flipped.dat"]
labelNames = ["negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat"]
validImgs = ["negativesvalid.dat", "sample_lightvalid.dat", "sample_shadowvalid.dat"]
validLabels = ["negativesvalidLabels.dat", "sample_lightvalidLabels.dat", "sample_shadowvalidLabels.dat"]
imgWidth = 224
imgHeight = 224
BATCH_SIZE = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 1

def loadData():
	X, y = [], []

	for imgSet in imgNames:
		print("Now operating on imgSet: " + str(imgSet))
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

				#print("l: " + str(l))
				if len(l) > 0:
					#print("Appending 1")
					y.append(1)
				else:
					#print("Appending 0")
					y.append(0)

	return np.array(X), np.array(y) 

'''
This function loads all training and validation data.  It returns a dictionary
containing all the data and some helper variables.    
'''
def globalLoadData():
	# Acquire the training data and validation data:
	print("Now loading training data")
	trainingImgs, trainingLbls = loadData()
	print("Completed loading training data.  Now loading validation data") 
	validationImgs, validationLbls = loadValidationData()
	print("Completed loading validation data")

	return dict(
		# Need to cast all data with float32 for images and int32 for labels 
		X_train = theano.shared(lasagne.utils.floatX(trainingImgs)),
		y_train = T.cast(theano.shared(trainingLbls), 'int32'),
		X_valid = theano.shared(lasagne.utils.floatX(validationImgs)),
		y_valid = T.cast(theano.shared(validationLbls), 'int32'),
		X_test = theano.shared(lasagne.utils.floatX(validationImgs)),
		y_test = T.cast(theano.shared(validationLbls), 'int32'),
		num_examples_train = X_train.shape[0],
		num_examples_valid = X_valid.shape[0],
		num_examples_test = X_test.shape[0],
	)

'''
This function constructs a neural network model that closely replicates that 
of the ImageNet Architecture.  The model is as follows:
'''
def build_model():
	print("Now creating Neural Network")
	l_in = lasagne.layers.InputLayer(
		shape=(None, 3, imgHeight, imgWidth))
	l_conv1 = lasagne.layers.Conv2DLayer(
		l_in,
		num_filters=24,
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
		num_filters=64,
		filter_size=(5,5),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))	
	l_pool2 = lasagne.layers.MaxPool2DLayer(
		l_conv2, 
		pool_size=(3,3),
		stride=2)
	l_conv3 = lasagne.layers.Conv2DLayer(
		l_pool2,
		num_filters=96,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_conv4 = lasagne.layers.Conv2DLayer(
		l_conv3,
		num_filters=96,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_conv5 = lasagne.layers.Conv2DLayer(
		l_conv4, 
		num_filters=64,
		filter_size=(3,3),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_pool3 = lasagne.layers.MaxPool2DLayer(
		l_conv5, 
		pool_size=(3,3),
		stride=2)
	l_hidden1 = lasagne.layers.DenseLayer(
		l_pool3,
		num_units=256,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_dropout1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
	l_hidden2 = lasagne.layers.DenseLayer(
		l_dropout1,
		num_units=256,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeNormal(gain='relu'))
	l_dropout2 = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)
	l_output = lasagne.layers.DenseLayer(
		l_dropout2,
		num_units=1,
		nonlinearity=lasagne.nonlinearities.softmax)

	print("Successfully created neural network")
	return l_output

'''
This function creates several Theano symbolic functions to help with
training, predicting, and validating the neural network. 
'''
def create_iter_functions(dataset, output_layer, X_tensor_type=T.tensor4,
                          batch_size=BATCH_SIZE,
						  learning_rate=LEARNING_RATE, momentum=MOMENTUM):

	print("Now creating iterator functions")

	batch_index = T.iscalar('batch_index')
	X_batch = X_tensor_type('X')
	y_batch = T.ivector('y')
	batch_slice = slice(batch_index * batch_size,
	                    (batch_index+1)*batch_size) 
	print("batch_slice: " + str(batch_slice))

	print("dataset['X_train']: " + str(dataset['X_train'][batch_slice]))

	'''
	Objective function: Categorical Crossentropy allows neural network
	to classify objects into different categories.  For now we have
	1 category: the precached sample.		  
	'''
	objective = lasagne.objectives.Objective(output_layer, 
	            loss_function=lasagne.objectives.categorical_crossentropy)

	loss_train = objective.get_loss(X_batch, target=y_batch)
	loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

	# Prediction function:
	pred = T.round(lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
	axis=1)

	accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

	all_params = lasagne.layers.get_all_params(output_layer)
	updates = lasagne.updates.nesterov_momentum(
	    loss_train, all_params, learning_rate, momentum)

	iter_train = theano.function(
	    [batch_index], loss_train,
		updates=updates,
		givens={
			X_batch: dataset['X_train'][batch_slice],
			y_batch: dataset['y_train'][batch_slice],
		},
	)

	iter_valid = theano.function(
	    [batch_index], [loss_eval, accuracy],
		givens={
			X_batch: dataset['X_valid'][batch_slice],
			y_batch: dataset['y_valid'][batch_slice],
		},
	)

	iter_test = theano.function(
		[batch_index], [loss_eval, accuracy],
		givens={
			X_batch: dataset['X_test'][batch_slice],
			y_batch: dataset['y_test'][batch_slice],
		},
	)

	print("Successfully created iterator functions")
	return dict(
	    train=iter_train,
		valid=iter_valid,
		test=iter_test,
	)

# Training function:
def train(iter_funcs, dataset, batch_size=BATCH_SIZE):

	# Determine the number of training/validation batches:
	num_batches_train = dataset['num_examples_train'] // batch_size
	num_batches_valid = dataset['num_examples_valid'] // batch_size

	print("num_batches_train: " + str(num_batches_train))
	print("num_batches_valid: " + str(num_batches_valid))

	for epoch in itertools.count(1):
		batch_train_losses = []
		for b in range(num_batches_train):
			batch_train_loss = iter_funcs["train"](b)
			print("batch_train_loss: " + str(batch_train_loss))
			batch_train_losses.append(batch_train_loss)

		avg_train_loss = np.mean(batch_train_losses)

		batch_valid_losses = []
		batch_valid_accuracies = []

		for b in range(num_batches_valid):
			batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
			print("batch_valid_accuracy: " + str(batch_valid_accuracy))
			batch_valid_losses.append(batch_valid_loss)
			batch_valid_accuracies.append(batch_valid_accuracy)

		avg_valid_loss = np.mean(batch_valid_losses)
		avg_valid_accuracy = np.mean(batch_valid_accuracies)

		yield {
			'number': epoch,
			'train_loss': avg_train_loss,
			'valid_loss': avg_valid_loss,
			'valid_accuracy': avg_valid_accuracy,
		}
			

# Main function:
def main(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, momentum=MOMENTUM):
	# Disable Lasagne Warnings:
	warnings.filterwarnings('ignore', module='lasagne')

	# Create the neural network:
	output_layer = build_model() 	

	# Load the validation data:
	print("Now loading validation data")
	validX, validY = loadValidationData()  	 
	validX = validX.astype(theano.config.floatX)
	validY = validY.astype(np.int32)
	 
	print("Done loading validation data")

	# Global iterator functions:
	batch_index = T.iscalar('batch_index')
	#X_batch = X_tensor_type('X')
	X_batch = T.tensor4('X')
	y_batch = T.ivector('y')
	batch_slice = slice(batch_index * batch_size,
	                    (batch_index+1)*batch_size) 
	'''
	Objective function: Categorical Crossentropy allows neural network
	to classify objects into different categories.  For now we have
	1 category: the precached sample.		  
	'''
	objective = lasagne.objectives.Objective(output_layer, 
	            loss_function=lasagne.objectives.categorical_crossentropy)

	loss_train = objective.get_loss(X_batch, target=y_batch)
	loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

	# Prediction function:
	pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True),
	axis=1)

	accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

	all_params = lasagne.layers.get_all_params(output_layer)
	updates = lasagne.updates.nesterov_momentum(
	    loss_train, all_params, learning_rate, momentum)

	print("Successfully created iterator functions")

	for i in xrange(len(imgNames)):

		X, y = [], []

		print("Now opening image set: " + str(imgNames[i]) + " and image labels: " + str(labelNames[i]))  

		with open(path+imgNames[i], "rb") as dataX, open(path+labelNames[i], "rb") as dataY:
			
			for imgItr in pickle.load(dataX):
				X.append(imgItr.reshape(3, imgHeight, imgWidth))

			for lblItr in pickle.load(dataY):
				if len(lblItr) > 0:
					y.append(1)
				else:
					y.append(0)

		# Typecast the next set of training data and labels:
		X = np.array(X[:10])
		y = np.array(y[:10])
		
		# Construct the dataset and iterator functions: 
		dataset = dict(
			# Need to cast all data with float32 for images and int32 for labels 
			X_train = theano.tensor._shared(lasagne.utils.floatX(X)),
			y_train = T.cast(theano.tensor._shared(y), 'int32'),
			X_valid = theano.tensor._shared(lasagne.utils.floatX(validX)),
			y_valid = T.cast(theano.tensor._shared(validY), 'int32'),
			X_test = theano.tensor._shared(lasagne.utils.floatX(validX)),
			y_test = T.cast(theano.tensor._shared(validY), 'int32'),
			num_examples_train = X.shape[0],
			num_examples_valid = validX.shape[0],
			num_examples_test = validX.shape[0],
		)

		print("Done loading data")
		#iter_funcs = create_iter_functions(dataset, output_layer)	

		print("Now creating iterator functions")
		iter_train = theano.function(
		    [batch_index], loss_train,
			updates=updates,
			givens={
				X_batch: dataset['X_train'][batch_slice],
				y_batch: dataset['y_train'][batch_slice],
			},
		)

		iter_valid = theano.function(
		    [batch_index], [loss_eval, accuracy],
			givens={
				X_batch: dataset['X_valid'][batch_slice],
				y_batch: dataset['y_valid'][batch_slice],
			},
		)

		iter_test = theano.function(
			[batch_index], [loss_eval, accuracy],
			givens={
				X_batch: dataset['X_test'][batch_slice],
				y_batch: dataset['y_test'][batch_slice],
			},
		)

		iter_funcs = dict(
	    		train=iter_train,
			valid=iter_valid,
			test=iter_test,
		)		

		# Begin training:
	
		print("Now beginning training of neural network: ") 
		now = time.time()
		try:
			for epoch in train(iter_funcs, dataset):
				print("Epoch {} of {} tok {:.3f}s".format(
					epoch['number'], num_epochs, time.time() - now))
				now = time.time()
				print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))	
				print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
				print("  validation accuracy:\t\t{:.2f} %%".format(epoch['valid_accuracy'] * 100))
				if epoch['number'] >= num_epochs:
					break
		except KeyboardInterrupt:
			pass

	return output_layer

if __name__ == "__main__":
	main()
