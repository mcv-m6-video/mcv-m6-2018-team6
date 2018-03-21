#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 20:43:17 2018

@author: ferran
"""
from __future__ import division
import sys
import datetime
import imageio
import math
import task1
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
import functions
import os
import math
import task1
import matplotlib.pyplot as plt
import numpy as np
#from frameworkJFMV2 import MOG as g
#from frameworkJFMV2 import Original as o

def image_translate(curr_img,future_img):
    block_size=25
    area_size=20
#    motion_matrix=task1.compute_block_matching(future_img, curr_img,block_size,area_size)
    OFtest=compute_block_matching(curr_img,future_img, block_size,area_size)

#    x_blocks=motion_matrix.shape[0]
#    y_blocks=motion_matrix.shape[1] 
#    OFtest=task1.create_OFtest(future_img, motion_matrix, block_size, x_blocks, y_blocks)
#    x=OFtest.shape[0]
#    y=OFtest.shape[1]
#    flow=np.zeros([curr_img.size[0],curr_img.size[1]])
#    OFtest = cv2.calcOpticalFlowFarneback(curr_img, future_img,flow, pyr_scale=0.5, levels=3, winsize=30,iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
#    U0=int(np.mean(OFtest[int(0.5*x):int(0.75*x),int(0.05*y):int(0.25*y),0]))
#    U1=int(np.mean(OFtest[int(0.5*x):int(0.75*x),int(0.05*y):int(0.25*y),1]))
    U0= OFtest[210,80,0]
    U1=  OFtest[210,80,1]
    print U0
    print U1
    print ('space')
    num_rows, num_cols = future_img.shape[:2]
    translation_matrix = np.float32([ [1,0,-U0], [0,1,-U1] ])
    img_translation = cv2.warpAffine(future_img, translation_matrix, (num_cols, num_rows))
    return img_translation


curr_dir = '/home/ferran/Desktop/M6_project/week4/training/image_0/000045_10.png'
future_dir   = '/home/ferran/Desktop/M6_project/week4/training/image_0/000045_11.png'
gt_dir= '/home/ferran/Desktop/M6_project/week4/training/flow_noc/000045_10.png'

future_img = cv2.imread(future_dir,0)
curr_img = cv2.imread(curr_dir,0)
block_size=np.arange(10,51,10)
area_size=np.arange(1,26,5)
OFgt = cv2.imread(gt_dir,-1)
x=block_size.shape[0]
y=area_size.shape[0]
matrix_m=np.zeros([x,y])
matrix_p=np.zeros([x,y])
c=0
for b in block_size:
    v=0
    for a in area_size:   
        #matrix=np.array[]

        OFtest=calcOpticalFlowBM (curr_img, future_img, b, a)
        msen,pepn=task1.MSEN_PEPN(OFtest,OFgt)
        matrix_m[c,v]=msen
        matrix_p[c,v]=pepn
        v=v+1
    c=c+1
    

#plt.savefig('PR_gray_highway1.png') #40 30 50 60


plt.plot(area_size,matrix_p[0],label='Block size=10')
plt.plot(area_size,matrix_p[1],label='Block size=20')
plt.plot(area_size,matrix_p[2],label='Block size=30')
plt.plot(area_size,matrix_p[3],label='Block size=40')
plt.plot(area_size,matrix_p[4],label='Block size=50')
#plt.plot(area_size,matrix_p[5],label='Block size=50')
#plt.plot(area_size,matrix_p[6],label='Block size=60')
#plt.plot(area_size,matrix_p[7],label='Block size=70')
#plt.plot(area_size,matrix_p[8],label='Block size=40')
#plt.plot(area_size,matrix_p[9],label='Block size=45')
#plt.plot(area_size,matrix_p[10],label='Block size=50')
#plt.plot(area_size,matrix_p[11],label='Block size=55')
#plt.plot(area_size,matrix_p[12],label='Block size=25')
#plt.plot(area_size,matrix_m[6],label='Block size=30')
#plt.plot(recall_mop,precision_mop,label='With Moprhological filters | AUC=0,62')
plt.xlabel('Area of search')
plt.ylabel('PEPN')
plt.title('Frame 45')
plt.legend()
#plt.savefig('PR_gray_highway1.png')
plt.show()
print ('ok 1')

#task1.PlotOpticalFlow1(OFtest,OFgt)
plt.plot(area_size,matrix_m[0],label='Block size=10')
plt.plot(area_size,matrix_m[1],label='Block size=20')
plt.plot(area_size,matrix_m[2],label='Block size=30')
plt.plot(area_size,matrix_m[3],label='Block size=40')
plt.plot(area_size,matrix_m[4],label='Block size=50')
#plt.plot(area_size,matrix_m[5],label='Block size=50')
#plt.plot(area_size,matrix_m[6],label='Block size=60')
#plt.plot(area_size,matrix_m[7],label='Block size=70')
#plt.plot(area_size,matrix_m[10],label='Block size=50')
#plt.plot(area_size,matrix_m[11],label='Block size=55')
#plt.plot(area_size,matrix_p[12],label='Block size=25')
#plt.plot(area_size,matrix_m[6],label='Block size=30')
#plt.plot(recall_mop,precision_mop,label='With Moprhological filters | AUC=0,62')
plt.xlabel('Area of search')
plt.ylabel('MSEN')
plt.title('Frame 45')
plt.legend()
#plt.savefig('PR_gray_highway1.png')
plt.show()
print ('ok 2')


curr_dir = '/home/ferran/Desktop/M6_project/week4/training/image_0/000157_10.png'
future_dir   = '/home/ferran/Desktop/M6_project/week4/training/image_0/000157_11.png'
gt_dir= '/home/ferran/Desktop/M6_project/week4/training/flow_noc/000157_10.png'

future_img = cv2.imread(future_dir,0)
curr_img = cv2.imread(curr_dir,0)
block_size=np.arange(10,51,10)
area_size=np.arange(1,26,5)
OFgt = cv2.imread(gt_dir,-1)
x=block_size.shape[0]
y=area_size.shape[0]
matrix_m2=np.zeros([x,y])
matrix_p2=np.zeros([x,y])
c=0
for b in block_size:
    v=0
    for a in area_size:   
        #matrix=np.array[]
#        motion_matrix=task1.compute_block_matching(future_img, curr_img,b,a)
##        motion_matrix1=motion_matrix
#        
#        x_blocks=motion_matrix.shape[0]
#        y_blocks=motion_matrix.shape[1]
#        
#        OFtest=task1.create_OFtest(future_img, motion_matrix, b, x_blocks, y_blocks)
        OFtest=calcOpticalFlowBM (curr_img, future_img, b, a)
        
        msen,pepn=task1.MSEN_PEPN(OFtest,OFgt)
        matrix_m2[c,v]=msen
        matrix_p2[c,v]=pepn
        v=v+1
    c=c+1
    

#plt.savefig('PR_gray_highway1.png') #40 30 50 60


plt.plot(area_size,matrix_p2[0],label='Block size=10')
plt.plot(area_size,matrix_p2[1],label='Block size=20')
plt.plot(area_size,matrix_p2[2],label='Block size=30')
plt.plot(area_size,matrix_p2[3],label='Block size=40')
plt.plot(area_size,matrix_p2[4],label='Block size=50')
#plt.plot(area_size,matrix_p2[5],label='Block size=50')
#plt.plot(area_size,matrix_p2[6],label='Block size=60')
#plt.plot(area_size,matrix_p2[7],label='Block size=70')
#plt.plot(area_size,matrix_p[8],label='Block size=40')
#plt.plot(area_size,matrix_p[9],label='Block size=45')
#plt.plot(area_size,matrix_p[10],label='Block size=50')
#plt.plot(area_size,matrix_p[11],label='Block size=55')
#plt.plot(area_size,matrix_p[12],label='Block size=25')
#plt.plot(area_size,matrix_m[6],label='Block size=30')
#plt.plot(recall_mop,precision_mop,label='With Moprhological filters | AUC=0,62')
plt.xlabel('Area of search')
plt.ylabel('PEPN')
plt.title('Frame 157')
plt.legend()
#plt.savefig('PR_gray_highway1.png')
plt.show()
print ('ok 3')

#task1.PlotOpticalFlow1(OFtest,OFgt)
plt.plot(area_size,matrix_m2[0],label='Block size=10')
plt.plot(area_size,matrix_m2[1],label='Block size=20')
plt.plot(area_size,matrix_m2[2],label='Block size=30')
plt.plot(area_size,matrix_m2[3],label='Block size=40')
plt.plot(area_size,matrix_m2[4],label='Block size=50')
#plt.plot(area_size,matrix_m2[4],label='Block size=40')
#plt.plot(area_size,matrix_m2[5],label='Block size=50')
#plt.plot(area_size,matrix_m2[6],label='Block size=60')
#plt.plot(area_size,matrix_m2[7],label='Block size=70')
#plt.plot(area_size,matrix_m[10],label='Block size=50')
#plt.plot(area_size,matrix_m[11],label='Block size=55')
#plt.plot(area_size,matrix_p[12],label='Block size=25')
#plt.plot(area_size,matrix_m[6],label='Block size=30')
#plt.plot(recall_mop,precision_mop,label='With Moprhological filters | AUC=0,62')
plt.xlabel('Area of search')
plt.ylabel('MSEN')
plt.title('Frame 157')
plt.legend()
#plt.savefig('PR_gray_highway1.png')
plt.show()
print ('ok 4')

#########################


