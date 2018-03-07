#title           :main.py
#description     :MSc in Computer Vision
#author          :Juan Felipe Montesinos
#date            :02-march-2018
#version         :0.1
#usage           :python pyscript.py
#notes           :
#python_version  :2.7
#scikit-image    :0.13.1
#dlib            :19.9.0
#==============================================================================
import numpy as np
import cv2
import sys
import os
from frameworkJFM import MOG as g
from frameworkJFM import Original as o

#data_dir = '/media/jfm/Slave/Data/datasets/fall/input'
#gt_dir   = '/media/jfm/Slave/Data/datasets/fall/groundtruth'
#
#backgroundSubtractor = cv2.BackgroundSubtractorMOG(history=100, 
#                        nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
#frames_test = []
#gt_test = []
#frames_train = []
#gt_train = []
#for i in range(1460,1511):
#    frames_train.append('in00'+str(i)+'.jpg')
#    gt_train.append('gt00'+str(i)+'.png')
#
#for i in range(1511,1561):
#    frames_test.append('in00'+str(i)+'.jpg')
#    gt_test.append('gt00'+str(i)+'.png')
#    print 'in00'+str(i)+'.jpg loaded'
#print 'Done!'
#results_list_dir = []
#mog = g('MOG2',data_dir,gt_dir,'RGB')
#mog.get_1D(frames_train)
#for i in range(1511,1561):
#    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
#    image = cv2.imread(im_dir,-1)
#    fgmask = mog.get_motion(image,1)
#    cv2.imwrite('background/00'+str(i)+'.png',fgmask)
#    results_list_dir.append('background/00'+str(i)+'.png')
#
#f1  = mog.evaluateSeveralFrames(frames_test,gt_test,0.5)
#mog.errorPainting(frames_test,gt_test,results_list_dir)
#
#animar = o('highway_mog','background',gt_dir)
#res_list = []
#for i in range(1511,1561):
#    res_list.append('00'+str(i)+'.png')
#animar.animacion(res_list)
##### ===========================================================================
##### ===========================================================================
data_dir = '/media/jfm/Slave/Data/datasets/highway/input'
gt_dir   = '/media/jfm/Slave/Data/datasets/highway/groundtruth'

backgroundSubtractor = cv2.BackgroundSubtractorMOG(history=100, 
                        nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
frames_test = []
gt_test = []
frames_train = []
gt_train = []
for i in range(1050,1201):
    frames_train.append('in00'+str(i)+'.jpg')
    gt_train.append('gt00'+str(i)+'.png')

for i in range(1201,1351):
    frames_test.append('in00'+str(i)+'.jpg')
    gt_test.append('gt00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg loaded'
print 'Done!'
results_list_dir = []
mog = g('MOG2',data_dir,gt_dir,'RGB')
mog.get_1D(frames_train)
for i in range(1201,1351):
    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
    image = cv2.imread(im_dir,-1)
    fgmask = mog.get_motion(image,1)
    cv2.imwrite('background/00'+str(i)+'.png',fgmask)
    results_list_dir.append('background/00'+str(i)+'.png')

f1  = mog.evaluateSeveralFrames(frames_test,gt_test,0.5)
mog.errorPainting(frames_test,gt_test,results_list_dir)

animar = o('highway_mog','background',gt_dir)
res_list = []
for i in range(1201,1351):
    res_list.append('00'+str(i)+'.png')
animar.animacion(res_list)