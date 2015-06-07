#!/usr/bin/python

'''
rockie-svm.py - This script creates and trains a support vector machine
to recognize the precached sample for Phase 1.   

RPI Rock Raiders
5/6/15

Last Updated: Bryant Pong: 5/6/15 - 11:03 PM
'''

# Python Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import cPickle as pickle
import os
import gc

# Globals:
dataFolder = "../data/pickle/"
imgNames = ["negativesImgs1_flipped.dat", "sample_lightImgs1_flipped.dat", "sample_shadowImgs1_flipped.dat", \
"negativesImgs2.dat", "sample_lightImgs2.dat", "sample_shadowImgs2.dat", \
"negativesImgs2_flipped.dat", "sample_lightImgs2_flipped.dat", "sample_shadowImgs2_flipped.dat"]

lblNames = ["negativesLabels1.dat", "sample_lightLabels1.dat", "sample_shadowLabels1.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat", \
"negativesLabels2.dat", "sample_lightLabels2.dat", "sample_shadowLabels2.dat"]

'''
This function performs a sliding window technique to divide each image into 
regions to examine.
'''
def slidingWindow(img, labels, winLen, winHgt): 

	#print("labels: " + str(labels))

	# Size of the image:
	imgLen = img.shape[1]
	imgHgt = img.shape[0]

	# Determine if there is a sample in this image:
	sample = False 
	sampleMidpointCol, sampleMidpointRow = None, None
	if len(labels) > 0:
		sample = True
		avg = np.mean(labels, axis=0)
		sampleMidpointCol = avg[1]
		sampleMidpointRow = avg[0]

		#print("midpoint col (X): " + str(sampleMidpointCol))
		#print("midpoint row (Y): " + str(sampleMidpointRow))

	# Ensure that this image can be evenly divided by winLen, winHgt:
	if winLen == 0 or winHgt == 0 or imgLen % winLen != 0 or imgHgt % winHgt != 0:
		print("Error window parameters do not divide img evenly!")
		return []	
											
	# Determine the number of window blocks to cover:
	numBlkLen = int(imgLen / winLen)
	numBlkHgt = int(imgHgt / winHgt)

	#print("You can create: " + str(numBlkLen) + " blocks horizontally x " + \
	#str(numBlkHgt) + " blocks vertically")
																	
	imgs = []
	lbls = []

	# Extract the blocks:
	for m in xrange(numBlkHgt):
		for n in xrange(numBlkLen):
			nextCornerRow = m*winHgt
			nextCornerCol = n*winLen

			#print("next corner at (" + str(nextCornerCol) + "," + str(nextCornerRow) + ")")

			if sample:
				#print("sampleMidpointRow: " + str(sampleMidpointRow))
				#print("sampleMidpointCol: " + str(sampleMidpointCol))
				if sampleMidpointRow >= nextCornerCol and sampleMidpointRow <= nextCornerCol + winHgt and \
					sampleMidpointCol >= nextCornerRow and sampleMidpointCol <= nextCornerRow + winLen:
					lbls.append(1)
				else:
					lbls.append(-1)
			else:
				lbls.append(-1)

			imgs.append(img[nextCornerRow:nextCornerRow+winHgt, nextCornerCol:nextCornerCol+winLen])

	return imgs, lbls

'''
This function loads the training images and labels. 
'''
def loadData():

	for i in xrange(len(imgNames)):

		descriptors = []
		svmLabels = []

		with open(dataFolder+imgNames[i], "rb") as X, open(dataFolder+lblNames[i], "rb") as y:
			print("Now operating on dataset: " + imgNames[i])
			
			imgs = pickle.load(X)
			labels = pickle.load(y)

			for counter in xrange(len(imgs)):

				print("Now operating on image: " + str(counter) + " out of: " + str(len(imgs)))				
				# Display the images:
				#plt.imshow(imgs[counter])  
				#plt.show()

				# Run the sliding windowing algorithm.  Need to rescale the labels by a factor of 2.
				newWindows, newLabels = slidingWindow(imgs[counter], labels[counter]/2, 160, 160)

				'''
				for temp in xrange(len(newWindows)):
					print(newLabels[temp])
					plt.imshow(newWindows[temp])
					plt.show()
				'''

				# Calculate the descriptors:
				for imgCounter in xrange(len(newWindows)): 
					descriptors.append(calcDescriptor(newWindows[imgCounter]))
				svmLabels.extend(newLabels)

			print("Now dumping descriptors and svm labels for dataset: " + str(imgNames[i]))
			pickle.dump(descriptors, open(imgNames[i]+"descriptors.pkl", "wb"))
			pickle.dump(svmLabels, open(imgNames[i]+"svmdescriptors.pkl", "wb"))
			print("Done dumping")

		# Reduce fragmented memory space:
		print("Now garbage collecting")
		gc.collect()					

	return [], []

'''
This function creates a histogram-based image descriptor for the SVM to train
on.  The descriptor is a 8 x 8 x 8 color-based histogram.   
'''
def calcDescriptor(img):
	flattenedImg = img.flatten()

	numElements = 0
	nextPixel = []
	nextPos = 0

	convertedImg = np.zeros((160*160, 3))

	for val in np.nditer(flattenedImg):
		numElements += 1
		nextPixel.append(val)

		if numElements >= 3:
			convertedImg[nextPos] = nextPixel
			nextPos += 1

			numElements = 0
			nextPixel = []

	histogram, _ = np.histogramdd(convertedImg, bins=[8, 8, 8])

	return histogram.flatten()

# Main function.  Select either "train" or "predict"   
def main(action):

	if action == "train":
		# Train the support vector machine
		print("Now training support vector machine")	

		# Load the training images and labels:  
		print("Now loading training images and labels")
		X, y = loadData() 

if __name__ == "__main__":
	main("train")
