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


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)

    
"""
============================== this is the traffic dataset ===========================
"""
data_dir = '/home/ferran/Desktop/M6_project/datasets/traffic/input/'

frames_train = []


print 'Reading background modeling files:...'
for i in range(950,1000):
    frames_train.append('in000'+str(i)+'.jpg')
    print 'in000'+str(i)+'.jpg loaded'
for i in range(1000,1051):
    frames_train.append('in00'+str(i)+'.jpg')
    print 'in00'+str(i)+'.jpg loaded'

im_curr = os.path.join(data_dir, frames_train[0])
curr_img = cv2.imread(im_curr,-1)
curr_img = cv2.cvtColor(curr_img,cv2.COLOR_BGR2GRAY) 
results_list_dir=[]
trans_img=curr_img
n=0
for i in sorted(frames_train[0:101]):

    curr_img=trans_img  
    im_future = os.path.join(data_dir, frames_train[n])
    future_img = cv2.imread(im_future,-1)
    future_img = cv2.cvtColor(future_img,cv2.COLOR_BGR2GRAY)
    print im_future
    trans_img= image_translate(curr_img,future_img)
    cv2.imwrite('results_traffic_gray/00'+str(i)+'.png',trans_img)
    results_list_dir.append('results_traffic_gray/00'+str(i)+'.png')
    n=n+1 
    
    
create_gif(results_list_dir, 0.1)
create_gif(frames_train, 0.1)




