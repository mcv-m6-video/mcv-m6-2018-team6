#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:10:58 2018

@author: yixiao
"""

from __future__ import division
import sys
import datetime
import imageio
import math
import evaluation
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage import morphology
import cv2
import os
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support as PRFmetrics
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import binary_fill_holes, generate_binary_structure
import cv2
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
#from frameworkJFMV2 import MOG as g
#from frameworkJFMV2 import Original as o
def euclidianDistance(current_block, search_block):


    dist = sum(sum(pow(current_block-search_block,2)))

    return dist

def blockMatching(curr_img, toExplore_img, blockSize, areaOfSearch):

    nRows, nCols = curr_img.shape
    
    u = np.zeros([nRows,nCols])
    v = np.zeros([nRows,nCols])
    
    # Iterate through each block in the current image
    for colIdx in range(0, nCols, blockSize):
        for rowIdx in range(0, nRows, blockSize):
    
            curr_block = curr_img[rowIdx:rowIdx+blockSize, colIdx:colIdx+blockSize]
    
            dist_vec = []; v_vec = []; u_vec = []
    
            # For each block iterate through all possible matches on the searching image
            for colIdx_s in range(np.max((0, colIdx-areaOfSearch)), np.min((nCols-blockSize, colIdx+areaOfSearch))):
                for rowIdx_s in range(np.max((0, rowIdx-areaOfSearch)), np.min((nRows-blockSize, rowIdx+areaOfSearch))):
    
                    search_block = toExplore_img[rowIdx_s:rowIdx_s+blockSize, colIdx_s:colIdx_s+blockSize]
    
    # Save euclidian distance between current block and each possible correspondecen and their corresponding OF vector
                    dist_vec =np.append(dist_vec, euclidianDistance(curr_block, search_block))
                    u_vec = np.append(u_vec, colIdx_s - colIdx)
                    v_vec = np.append(v_vec, rowIdx_s - rowIdx)
    
    # Get the OF vector corresponding to the more similar patch
            min, max, minLoc, maxLoc = cv2.minMaxLoc(dist_vec)
            u[rowIdx:rowIdx+blockSize, colIdx:colIdx+blockSize] = u_vec[minLoc[1]]
            v[rowIdx:rowIdx + blockSize, colIdx:colIdx + blockSize] = v_vec[minLoc[1]]
    
    return u, v

def calcOpticalFlowBM (curr_img, toExplore_img, blockSize, areaOfSearch):

    OF_image = np.zeros([curr_img.shape[0], curr_img.shape[1], 3])
    # Number of blocks in each dimension given img size and blockSize
    x_blocks = int(np.ceil(toExplore_img.shape[0] / blockSize))
    y_blocks = int(np.ceil(toExplore_img.shape[1] / blockSize))

    
    # Add padding in both imageS
    curr_img_pad = np.zeros([x_blocks * blockSize, y_blocks * blockSize])
    toExplore_img_pad = np.zeros([x_blocks * blockSize, y_blocks * blockSize])
    curr_img_pad[0:toExplore_img.shape[0], 0:toExplore_img.shape[1]] = curr_img[:,:]
    toExplore_img_pad[0:toExplore_img.shape[0], 0:toExplore_img.shape[1]] = toExplore_img[:,:]
    
    # Block Matching search
    u, v = blockMatching(curr_img_pad, toExplore_img_pad, blockSize, areaOfSearch)
    
    OF_image[:, :, 0] = u[:curr_img.shape[0], :curr_img.shape[1]]
    OF_image[:, :, 1] = v[:curr_img.shape[0], :curr_img.shape[1]]
    
    return OF_image

def image_translate(curr_img,future_img):
    block_size=25
    area_size=20
#    motion_matrix=task1.compute_block_matching(future_img, curr_img,block_size,area_size)
    OFtest=calcOpticalFlowBM(curr_img,future_img, block_size,area_size)

#    x_blocks=motion_matrix.shape[0]
#    y_blocks=motion_matrix.shape[1] 
#    OFtest=task1.create_OFtest(future_img, motion_matrix, block_size, x_blocks, y_blocks)
#    x=OFtest.shape[0]
#    y=OFtest.shape[1]
#    flow=np.zeros([curr_img.size[0],curr_img.size[1]])
#    OFtest = cv2.calcOpticalFlowFarneback(curr_img, future_img,flow, pyr_scale=0.5, levels=3, winsize=30,iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
#    U0=int(np.mean(OFtest[int(0.5*x):int(0.75*x),int(0.05*y):int(0.25*y),0]))
#    U1=int(np.mean(OFtest[int(0.5*x):int(0.75*x),int(0.05*y):int(0.25*y),1]))
    U0= OFtest[267,25,0]
    U1=  OFtest[267,25,1]
    print U0
    print U1
    print ('space')
    num_rows, num_cols = future_img.shape[:2]
    translation_matrix = np.float32([ [1,0,-U0], [0,1,-U1] ])
    img_translation = cv2.warpAffine(future_img, translation_matrix, (num_cols, num_rows))
    return img_translation