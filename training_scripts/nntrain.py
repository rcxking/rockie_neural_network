#!/usr/bin/python

'''
nntrain.py - Training script for the precached sample.

RPI Rock Raiders
5/18/15

Last Updated: Bryant Pong: 5/18/15 - 9:35 PM     
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

	# A list of pickle files to open:
	fileNames = [folderName+"samplelight1.dat", folderName+"samplelight2.dat", folderName+"negatives1.dat", folderName+"negatives2.dat"] 

	for fileName in fileNames:

		with open(fileName, "rb") as f:

			print("Now processing: " + str(fileName))

			data = pickle.load(f)

			for img in data:
				imgs.append(cv2.resize(img[0], (360, 240)))
				if len(img[0][1]) == 0:
					labels.append(0)
				else:
					labels.append(1)

			print("Done processing: " + str(fileName))

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
