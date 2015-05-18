#!/usr/bin/python

import pygame
from pygame.locals import *
import numpy as np
import sys
import os
import cv2
import cPickle as pickle

pygame.init()

def points_of_interest(cv_image):
  image = pygame.image.frombuffer(cv_image, (cv_image.shape[1], cv_image.shape[0]), 'RGB')

  size = image.get_size()
  screen = pygame.display.set_mode(size)
  points = np.zeros((0,2), dtype='uint32')

  running = True
  pressed = False
  clock = pygame.time.Clock()
  while running:
    screen.fill((255,255,255))
    screen.blit(image, (0,0))
    pygame.display.flip()
    for event in pygame.event.get():
      if event.type == QUIT:
        running = False
      if event.type == KEYDOWN:
        running = False
      if event.type == MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0]:
        pos = pygame.mouse.get_pos()
        print "Point:", pos
        points = np.vstack((points, np.array(list(pos), dtype='uint32')), )
    clock.tick(30)
  print "Points:", points
  return points


def extract_points(file_name):    
  image = cv2.cvtColor(file_name, cv2.COLOR_BGR2RGB)
  points = points_of_interest(image)

  return points

