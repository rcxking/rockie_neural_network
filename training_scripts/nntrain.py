#!/usr/bin/python

'''
nntrain.py - Training script for the precached sample.

RPI Rock Raiders
5/18/15

Last Updated: Bryant Pong: 5/18/15 - 7:43 PM     
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

'''
This function loads in the training data of images.  It resizes the images
(which are initially  
'''
def loadData():
	
	# The training images and labels:
	imgs, labels = [], []		

	folderName = "/home/bryant/rockie_neural_network/data/pickle/"
	#data1 = pickle.load(open(folderName+"sample_light.dat", "rb"))
	#data2 = pickle.load(open(folderName+"negatives.dat", "rb"))


	with open(folderName+"sample_light.dat", "rb") as f:

		data1 = pickle.load(f)

		for img in data1:
			imgs.append(img[0])
			if len(img[0][1]) == 0:
				labels.append(0)
			else:
				labels.append(1)

	with open(folderName+"negatives.dat", "rb") as f:

		data2 = pickle.load(f)

		for img in data2:
			imgs.append(img[0])
			if len(img[0][1]) == 0:
				labels.append(0)
			else:
				labels.append(1)

	return np.array(imgs).astype(theano.config.floatX), np.array(labels).astype(np.int32)
	
# Training function:
def train():
	
	# Deactivate warnings for lasagne:
	warnings.filterwarnings("ignore", module="lasagne")

	# Load the training data of the precached sample: 
	print("Now loading training data")
	X, y = loadData()
	print("Complete loading training data")

	# Create the 

	print("Now beginning to train convolutional neural network")

# Main function runner:
if __name__ == "__main__":
	train()
