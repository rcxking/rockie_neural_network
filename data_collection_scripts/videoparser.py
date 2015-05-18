#!/usr/bin/python

'''
videoparser.py - A helper script that extracts image frames from a video via
OpenCV.

pickles a list of tuples of frames and points of interest in a file with the
extension .dat in the same folder as the video file

Computational Vision Final Project
Bryant Pong/Micah Corah
CSCI-4962
4/24/15

Last Updated: Mica Cora - 5/3/15
'''

# Python Imports:
import cv2
import numpy as np
import os
import sys
import pickle
from points_of_interest import points_of_interest

# Main function:
def processVideo(video):
  print "Begin processing", video
  file_name, extension = os.path.splitext(video)
  data_file_name = file_name + ".dat"

  # Check if the data file to write to already exists and if so ask whether to
  # overwrite
  if os.path.isfile(data_file_name):
    print(data_file_name + " already exists!")
    overwrite = raw_input("Data file already exists. Overwrite? Y/N: ")
    if not overwrite.lower()[0] == 'y':
      return

  # Load the video:
  vid = cv2.VideoCapture(video)

  # Write only 1 image every 10 frames:
  num_frames = 0

  frames_points = []

  while(vid.isOpened()):

    ret, frame = vid.read()
    if not ret:
      break

    num_frames += 1

	# Capture every 5 frames:
    if num_frames % 5 == 0: 
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      points = points_of_interest(frame)

      frames_points.append((frame, points))

  vid.release()
  cv2.destroyAllWindows()

  data_file = open(data_file_name, 'w')
  print "Pickling", video, "to", data_file_name
  pickle.dump(frames_points, data_file)
  print "Done processing", video

# Main function runner.  Pass in the path to the video you wish to parse:
if __name__ == '__main__':

  if len(sys.argv) < 2:
    print("Usage: videoparser.py <src video> <dst folder>")  
  else:
    for arg in sys.argv[1:]:
      processVideo(arg) 
