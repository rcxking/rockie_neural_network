#!/usr/bin/python

'''
data_augmentation.py - This script augments data by performing transformations

RPI Rock Raiders
6/3/15

Last Updated: Bryant Pong: 6/3/15 - 3:07 PM    
'''

# Python Imports:
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pickle

dataFolder = "../data/pickle/"
dataFiles = ["negativesImgs1", "negativesImgs2", "sample_lightImgs1", "sample_lightImgs2", "sample_shadowImgs1", "sample_shadowImgs2"]

def main():

	for fileName in dataFiles:
	
		print("Now flipping: " + fileName)
		nextDataset = []

		with open(dataFolder+fileName+".dat", "rb") as f:
			for img in pickle.load(f):
				nextDataset.append(cv2.flip(img, 1))		

				#plt.imshow(cv2.flip(img, 1))
				#plt.show()

		pickle.dump(nextDataset, open(fileName+"_flipped.dat", "wb"))

	print("All done")

if __name__ == "__main__":
	main()
