#!/usr/bin/python

'''
splitdata.py - This script opens up the three large pickled data files,
resizes the images, and splits the each data set into 3 sets.

RPI Rock Raiders
5/27/15

Last Updated: Bryant Pong: 5/27/15 - 4:33 PM      
'''

# Python Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pickle

dataFolder = "../data/pickle/" 
dataNames = ["sample_shadow"]
padding = np.zeros((80, 184, 3))

# Main function:
def main():
	
	# Open each file:
	for filename in dataNames:
		with open(dataFolder+filename+".dat", "rb") as f:
			'''
			Create three datasets for each file:
			1) First half of the data
			2) Second half of the data
			3) The last 100 samples as a validation set     
			'''
			print("Now processing " + str(filename)+".dat")
			firstHalfImgs, secondHalfImgs, validImgs = [], [], [] 
			firstHalfLabels, secondHalfLabels, validLabels = [], [], []

			curDataset = pickle.load(f)
			last100 = len(curDataset) - 100	  		
			midpoint = (len(curDataset) - 100) / 2

			for i in xrange(0, midpoint):
				nextImg = curDataset[i][0]
				nextImg = cv2.resize(nextImg, (184, 104))	 
				paddedImg = np.vstack([nextImg, padding]).astype(np.uint8)
				firstHalfImgs.append(paddedImg)
				firstHalfLabels.append(curDataset[i][1])
			for i in xrange(midpoint, last100):
				nextImg = curDataset[i][0]
				nextImg = cv2.resize(nextImg, (184, 104))
				paddedImg = np.vstack([nextImg, padding]).astype(np.uint8)
				secondHalfImgs.append(paddedImg)
				secondHalfLabels.append(curDataset[i][1])
			for i in xrange(last100, last100+100):
				nextImg = curDataset[i][0]
				nextImg = cv2.resize(nextImg, (184, 104))
				paddedImg = np.vstack([nextImg, padding]).astype(np.uint8)
				validImgs.append(paddedImg)
				validLabels.append(curDataset[i][1])

			print("Now writing data")
			pickle.dump(firstHalfImgs, open(filename+"Imgs1.dat", "wb"))
			pickle.dump(secondHalfImgs, open(filename+"Imgs2.dat", "wb"))
			pickle.dump(validImgs, open(filename+"valid.dat", "wb"))
			pickle.dump(firstHalfLabels, open(filename+"Labels1.dat", "wb"))
			pickle.dump(secondHalfLabels, open(filename+"Labels2.dat", "wb"))
			pickle.dump(validLabels, open(filename+"validLabels.dat", "wb"))

				
	print("All done")

if __name__ == "__main__":
	main()
