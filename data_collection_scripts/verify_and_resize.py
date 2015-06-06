#!/usr/bin/python

'''
verify_and_resize.py - This script opens up pickled data and resizes the     
images to 184 x 104.  Then 80 rows of zeros is used to pad up the image
such that the resulting image is 184 x 184 px.

RPI Rock Raiders
5/26/15

Last Updated: Bryant Pong: 6/5/15 - 4:55 PM
'''

# Python Imports:
import cv2
import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt  

# Pickled data:
folder = "../data/pickle/"
filenames = ["sample_lightImgs1.dat", "sample_lightImgs2.dat", "negativesImgs1.dat", "negativesImgs2.dat", "sample_shadowImgs1.dat", "sample_shadowImgs2.dat"]

def opendata():
	
	imgs = []
	labels = []

	for filename in filenames:
		with open(folder+filename) as f:
			
			for dataTuple in pickle.load(f)
				 nextImg = dataTuple[0]
				 nextImg = cv2.resize(nextImg, (112, 112))
				 imgs.append(nextImg)
				 labels.append(dataTuple[1])			 	 			

	return imgs, labels

def main():
	print("Now loading training data and labels: ")
	X, y = opendata()
	print("Done resizing data")

if __name__ == "__main__":
	main()
