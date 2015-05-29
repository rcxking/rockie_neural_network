#!/usr/bin/python

'''
verify_and_resize.py - This script opens up pickled data and resizes the     
images to 184 x 104.  Then 80 rows of zeros is used to pad up the image
such that the resulting image is 184 x 184 px.

RPI Rock Raiders
5/26/15

Last Updated: Bryant Pong: 5/26/15 - 4:31 PM 
'''

# Python Imports:
import cv2
import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt  

# Pickled data:
folder = "../data/pickle/"
filenames = ["samplelight1.dat", "samplelight2.dat", "negatives1.dat", "negatives2.dat"]

def opendata():
	
	imgs = []
	labels = []

	for filename in filenames:
		with open(folder+filename) as f:
			nextData = pickle.load(f)

			for dataTuple in nextData:
				 nextImg = dataTuple[0]
				 nextImg = cv2.resize(nextImg, (224, 224))
				 imgs.append(nextImg)
				 labels.append(dataTuple[1])			 	 			

	return imgs, labels

def main():
	print("Now loading training data and labels: ")
	X, y = opendata()
	print("Done loading training data and labels")

	print("Now damping data and labels")
	pickle.dump(X, open("X.dat", "wb"))
	pickle.dump(y, open("y.dat", "wb"))
	print("Done dumping data")

if __name__ == "__main__":
	main()
