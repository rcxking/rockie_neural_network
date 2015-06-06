#!/usr/bin/python

'''
nntrain.py - Training script for a neural network to recognize the precached
sample.

RPI Rock Raiders
5/27/15

Last Updated: Bryant Pong: 6/5/15 - 9:19 PM
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
"negativesImgs2_flipped.dat", "sample_lightImgs2_flipped.dat", "sample_shadowImgs2_flipped.dat"]
labelNames = ["negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat"]
validImgs = ["negativesvalid.dat", "sample_lightvalid.dat", "sample_shadowvalid.dat"]
validLabels = ["negativesvalidLabels.dat", "sample_lightvalidLabels.dat", "sample_shadowvalidLabels.dat"]
imgWidth = 112
imgHeight = 112
BATCH_SIZE = 100
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
NUM_EPOCHS = 200

def loadData():
	X, y = [], []

	for imgSet in imgNames:
		print("Now operating on imgSet: " + str(imgSet))
		with open(path+imgSet, "rb") as f:
			for i in pickle.load(f):
				#plt.imshow(i)
				#plt.show()
				#resizedImg = i.reshape(3, imgHeight, imgWidth)
				resizedImg = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).reshape(1, imgHeight, imgWidth)
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
				#resizedImg = i.reshape(3, imgHeight, imgWidth)
				resizedImg = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).reshape(1, imgHeight, imgWidth)
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

def loadGlobalData():
	X,y = [],[]

	# Load the training data:
	print("Now loading training data")
	X_train, y_train = loadData()
	print("Done loading training data.  Now loading validation data")
	X_valid, y_valid = loadValidationData()
	print("Done loading validation data.")

	print("X_train.shape: " + str(X_train.shape))
	print("X_valid.shape: " + str(X_valid.shape))

	return dict(
		X_train = theano.tensor._shared(lasagne.utils.floatX(X_train)),
		y_train = T.cast(theano.tensor._shared(y_train), 'int32'),
		X_valid = theano.tensor._shared(lasagne.utils.floatX(X_valid)),
		y_valid = T.cast(theano.tensor._shared(y_valid), 'int32'),
		X_test = theano.tensor._shared(lasagne.utils.floatX(X_valid)),	
		y_test = T.cast(theano.tensor._shared(y_valid), 'int32'),
		num_examples_train = X_train.shape[0],
        num_examples_valid = X_valid.shape[0],
        num_examples_test = X_valid.shape[0],
	)

'''
This function constructs a neural network model that closely replicates that 
of the ImageNet Architecture.  The model is as follows:
'''
def build_model(batch_size=BATCH_SIZE):
	print("Now creating Neural Network")
	l_in = lasagne.layers.InputLayer(
		shape=(None, 1, 112, 112))
	l_conv1 = lasagne.layers.Conv2DLayer(
		l_in,
		num_filters=48,
		filter_size=(5,5),
		stride=2,
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
		num_units=2,
		nonlinearity=lasagne.nonlinearities.softmax)

	print("Successfully created neural network")
	return l_output

def create_iter_functions(dataset, output_layer, X_tensor_type=T.ftensor4,
                          batch_size=BATCH_SIZE,
						  learning_rate=LEARNING_RATE,
						  momentum=MOMENTUM):

	batch_index = T.iscalar('batch_index')
	X_batch = X_tensor_type('x')
	y_batch = T.ivector('y')
	batch_slice = slice(batch_index * batch_size, (batch_index+1)*batch_size)

	objective = lasagne.objectives.Objective(output_layer,
		loss_function = lasagne.objectives.categorical_crossentropy)

	loss_train = objective.get_loss(X_batch, target=y_batch)
	loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

	pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True), axis=1)
	accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

	all_params = lasagne.layers.get_all_params(output_layer)
	updates = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)

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

	# Load all data:
	print("Now loading all data")
	dataset = loadGlobalData()
	print("Done loading all data")

	print("Now creating iterator functions")
	iter_funcs = create_iter_functions(dataset, output_layer) 
	print("Successfully created iterator functions")

	# Begin training:	
	print("Now beginning training of neural network: ") 
	now = time.time()

	'''
	These variables keep track of the lowest validation loss.      
	'''
	lowestValidationLoss = 9001.0

	try:
		for epoch in train(iter_funcs, dataset):
			print("Epoch {} of {} tok {:.3f}s".format(
				epoch['number'], num_epochs, time.time() - now))
			now = time.time()
			print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))	
			print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
			print("  validation accuracy:\t\t{:.2f} %%".format(epoch['valid_accuracy'] * 100))

			if float(epoch['valid_loss']) < lowestValidationLoss:
				print("This is our lowest validation loss!")
				best_params = lasagne.layers.get_all_param_values(output_layer)	 
				pickle.dump(best_params, open("best_params.pkl", "wb"))
				pickle.dump(lowestValidationLoss, open("lowestValidationLoss.pkl", "wb"))

			if epoch['number'] >= num_epochs or float(epoch['valid_loss']) > lowestValidationLoss:
				break
	except KeyboardInterrupt:
		pass

	return output_layer

if __name__ == "__main__":
	main()
