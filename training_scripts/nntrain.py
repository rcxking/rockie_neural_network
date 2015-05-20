#!/usr/bin/python

'''
nntrain.py - Training script for the precached sample.

RPI Rock Raiders
5/18/15

Last Updated: Bryant Pong: 5/19/15 - 10:43 PM     
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
This function dumps the training data of images into chunks.  It resizes the images
(which are initially  
'''
def dumpData():
	
	# The training images and labels:
	imgs, labels = [], []		

	folderName = "/home/bryant/rockie_neural_network/data/pickle/"
	#data1 = pickle.load(open(folderName+"sample_light.dat", "rb"))
	#data2 = pickle.load(open(folderName+"negatives.dat", "rb"))

	# A list of pickle files to open:
	fileNames = ["samplelight1.dat", "samplelight2.dat", "negatives1.dat", "negatives2.dat"] 

	for fileName in fileNames:

		with open(folderName+fileName, "rb") as f:

			print("Now processing: " + str(fileName))

			data = pickle.load(f)

			for img in data:
				imgs.append(cv2.resize(img[0], (360, 240)))
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
	
# Training function:
def train():
	
	# Deactivate warnings for lasagne:
	warnings.filterwarnings("ignore", module="lasagne")

	# Load the training data of the precached sample: 
	print("Now loading training data")
	dumpData()
	print("Complete loading training data")

	# Create the 

	print("Now beginning to train convolutional neural network")

# Main function runner:
if __name__ == "__main__":
	train()
