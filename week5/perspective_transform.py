#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 22:59:39 2018

@author: yixiao
"""

import perspective
import numpy as np
import cv2

# load the notecard code image, clone it, and initialize the 4 points
# that correspond to the 4 corners of the notecard
frame = cv2.imread("/home/yixiao/Documents/M6_project/week5/images/in000091.jpg")
clone = frame

# pts are the location of 4 points of the bounding box
pts = np.array([(240,0), (38, 327), (533,327), (350, 0)])

# loop over the points and draw them on the cloned image
for (x, y) in pts:
    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)

# apply the four point tranform to obtain a "birds eye view" of
# the notecard
warped, M = perspective.four_point_transform(frame, pts)

# show the original and warped images
cv2.imshow("Original", clone)
cv2.imshow("Warped", warped)
cv2.waitKey(0)