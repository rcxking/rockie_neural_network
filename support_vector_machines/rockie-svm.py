#!/usr/bin/python

'''
rockie-svm.py - This script creates and trains a support vector machine
to recognize the precached sample for Phase 1.   

RPI Rock Raiders
5/6/15

Last Updated: Bryant Pong: 5/7/15 - 2:30 PM
'''

# Python Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import confusion_matrix
import cPickle as pickle
import os
import gc

# Globals:
dataFolder = "../data/pickle/"
imgNames = ["sample_shadowImgs1.dat", \
"negativesImgs2.dat", "sample_lightImgs2.dat", "sample_shadowImgs2.dat"]
lblNames = ["sample_shadowLabels1.dat", \
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
				plt.imshow(imgs[counter])  
				plt.show()

				# Run the sliding windowing algorithm.  Need to rescale the labels by a factor of 2.
				newWindows, newLabels = slidingWindow(imgs[counter], labels[counter]/2, 160, 160)

				for temp in xrange(len(newWindows)):
					print(newLabels[temp])
					plt.imshow(newWindows[temp])
					plt.show()

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

# Training function:
def train(images, labels):

	print("type of labels: " + str(type(labels)))
	print("labels: " + str(labels))

	tempImgs = [i.tolist() for i in images]

	# Create the LinearSVM object:
	clf = svm.LinearSVC(C=1.0)

	X = np.array(tempImgs)

	# Begin classification:

	print("Now training support vector machine") 
	clf.fit(X, labels)
	print("Done training support vector machine")

	return clf

def loadTrainingData():
	trainFolder = "../data/svm_pickle/"
	trainImgs = ["lightImgs1Des", "lightImgs2Des", \
	"negImgs1Des", "negImgs2Des", \
	"shadImgs1Des", "shadImgs2Des"]
	trainLbls = ["lightImgs1Lbls", "lightImgs2Lbls", \
	"negImgs1Lbls", "negImgs2Lbls", \
	"shadImgs1Lbls", "shadImgs2Lbls"]

	allX, allY = [], []

	for i in xrange(len(trainImgs)):

		with open(trainFolder+trainImgs[i]+".pkl", "rb") as X, \
		     open(trainFolder+trainLbls[i]+".pkl", "rb") as y:
			
			allX.extend(pickle.load(X))
			allY.extend(pickle.load(y))

		gc.collect()

	return allX, allY
# Main function.  Select either "train" or "predict"   
def main(action):

	if action == "getdata":
		# Train the support vector machine
		print("Now training support vector machine")	

		# Load the training images and labels:  
		print("Now loading training images and labels")
		X, y = loadData()
	elif action == "train":
		print("Now training support vector machine")
		
		# Load the training images and labels:
		print("Now loading training images and labels")		 
		X, y = loadTrainingData()
		print("Done loading training images and labels")

		print("Now training SVM")
		SVM = train(X, y)
		print("Done training SVM")
		print("Now dumping SVM:")
		pickle.dump(SVM, open("SVM.pkl", "wb"))
		print("Done dumping SVM")

		print("All done")
	else:
		print("Now validating Support Vector Machine")

		print("Now loading SVM")
		svm = pickle.load(open("SVM.pkl", "rb")) 
		print("Done loading SVM.  Now loading validation images")
		sampleLight = pickle.load(open("../data/pickle/sample_lightImgs1.dat", "rb"))
		print("Done loading validation images.  Now loading labels") 
		sampleLightLabels = pickle.load(open("../data/pickle/sample_lightLabels1.dat"))
		print("Beginning validation")

		numCorrect = 0	
		counter = 0
		y_pred = []
		y_act = []
		for temp in sampleLight[:50]:
	
				
			#plt.imshow(temp)
			#plt.show()

			# Run the sliding window algorithm:
			slidingWindows, newLabels = slidingWindow(temp, sampleLightLabels[counter]/2, 160, 160)
			for i in xrange(len(slidingWindows)):
				testDes = calcDescriptor(slidingWindows[i])
				print("Prediction: " + str(svm.predict(testDes)[0]))
				#print("sampleLightLabel[counter]: " + str(sampleLightLabels[counter]))
				y_pred.append(svm.predict(testDes)[0])
				y_act.append(newLabels[i])
				if svm.predict(testDes)[0] == newLabels[i]:
					numCorrect += 1
				

				#plt.imshow(slidingWindows[i])
				#plt.show()
			counter += 1

		cMatrix = confusion_matrix(y_act, y_pred)
		print("Accuracy: " + str(float(numCorrect) / (50*12)))
		print("Confusion Matrix: " + str(cMatrix))
		print("All done")

if __name__ == "__main__":
	main("train")
	main("validate")
