#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:54:17 2018

@author: yixiao
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
#--------------------------------------------------------------------
# M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring
# Team06: YI XIAO, Juan Felipe Montesinos,Ferran Carrasquer

# week 1: Introduction, Databases, Evaluation Metrics

# This .py file is for defining some functions of evaluation
#--------------------------------------------------------------------
"""
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as PRFmetrics
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import math

# Input the folder path of the results and groundtruth, this function will output the precision, recall, f1_score for all the frames
def evaluateAllFrames(folder_path, gt_path):
    predVector = []
    trueVector = []
    if len(os.listdir(folder_path))==len(os.listdir(gt_path)):
        for filename in sorted(glob.glob(os.path.join(folder_path, 'test_*_00*.png'))):
            image = cv2.imread(filename,0)
            for ridx in range(image.shape[0]):
                for cidx in range(image.shape[1]):
                    predVector.append(image[ridx,cidx])
        for filename in sorted(glob.glob(os.path.join(gt_path, 'gt00*.png'))):
            gtImage = cv2.imread(filename,0)
            for ridx in range(gtImage.shape[0]):
                for cidx in range(gtImage.shape[1]):
                    trueVector.append(gtImage[ridx,cidx])
    else:
        print('The number of images are not corresponding!')
    trueArray = np.asarray(trueVector)
    predArray = np.asarray(predVector)        
    for j in range(len(trueArray)):
        if trueArray[j]==255:
            trueArray[j] = 1
        else:
            trueArray[j] = 0
    precision, recall,f1_score,support = PRFmetrics(trueArray, predArray, average='binary')
    return precision, recall, f1_score
    


# Input the pair of image and gt, this function will output the TP, FP, TN, FN
def evaluateOneFrame(frame,gt):
    predVector = []
    trueVector = []
    for ridx in range(frame.shape[0]):
        for cidx in range(frame.shape[1]):
            predVector.append(frame[ridx,cidx])
    for ridx in range(gt.shape[0]):
        for cidx in range(gt.shape[1]):
            trueVector.append(gt[ridx,cidx])
    trueArray = np.asarray(trueVector)
    predArray = np.asarray(predVector)     
    for i in range(len(trueArray)):
        if trueArray[i] == 255:
            trueArray[i] = 1
        else:
            trueArray[i] = 0
    _, _,f1_score_one,_ = PRFmetrics(trueArray, predArray, average='binary')  
    TP=0
    TN=0
    FP=0
    FN=0
    # for the gt, we only consider two classes(0,255) represent background and motion respectively.
    for j in range(len(trueArray)):
        # True Positive (TP): we predict a label of 255 is positive, and the gt is 255.
        if trueArray[j] == predArray[j] == 1:
            TP = TP+1
        # True Negative (TN): we predict a label of 0 is negative, and the gt is 0.
        if trueArray[j] == predArray[j] == 0:
            TN = TN+1
        # False Positive (FP): we predict a label of 255 is positive, but the gt is 0.
        if trueArray[j] ==0 and predArray[j] == 1:
            FP = FP+1
        # False Negative (FN): we predict a label of 0 is negative, but the gt is 255.
        if trueArray[j] == 1 and predArray[j] == 0:
            FN = FN+1      
    return TP, FN, f1_score_one
          

# This function is for evaluating the True Positive along time
def temproalTP(folder_path,gt_path):
    TF_list = []
    TP_list = []
    for filename in sorted(glob.glob(os.path.join(folder_path, 'test_*_00*.png'))):
        frame = cv2.imread(filename,0)
        num = str(filename[-8:-4])
        gtpath=os.path.join(gt_path, 'gt00'+num+'.png')
        gtImage = cv2.imread(gtpath,0)
        TP, FN, f1_score_frame = evaluateOneFrame(frame, gtImage)
        TotalForeground = TP + FN
        TP_list.append(TP)
        TF_list.append(TotalForeground)
    plt.figure(1)
    plt.plot(np.arange(len(TP_list)),TP_list,'b',label='Ture Positive')
    plt.plot(np.arange(len(TF_list)),TF_list,'r',label = 'Total Background')
    plt.xlabel("Frames")
    plt.ylabel("Number of pixels")
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12}
    plt.title('TP&TF VS frames', fontdict=font)
    plt.text(100, 7000, 'TP-blue', fontdict=font)
    plt.text(100, 6000, 'TF-red', fontdict=font)
    plt.show()

# This function is for evaluating the F1-score along time
def temproalFscore(folder_path,gt_path):
    f1score_list = []
    for filename in sorted(glob.glob(os.path.join(folder_path, 'test_*_00*.png'))):
        frame = cv2.imread(filename,0)
        num = str(filename[-8:-4])
        gtpath=os.path.join(gt_path, 'gt00'+num+'.png')
        gtImage = cv2.imread(gtpath,0)
        _, _, f1_score_frame = evaluateOneFrame(frame, gtImage)
        f1score_list.append(f1_score_frame)
    plt.ylim(0.00,1.00)
    plt.plot(np.arange(len(f1score_list)),f1score_list,'b',label = 'f1-score')
    plt.xlabel("Frames")
    plt.ylabel("precentage")
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12}
    plt.title('F1-score vs frames', fontdict=font)
    plt.show()
    
    
# This function is for calculating the mean square of error in Non-occluded areas
# and the percentage of Erroneous Pixels in Non-occluded areas 
def MSEN_PEPN(OFtest,OFgt):
    SquareErrorList=[]
    test_u=[]
    test_v=[]
    gt_u=[]
    gt_v=[]
    gt_valid=[]
    for ridx in range(OFtest.shape[0]):
        for cidx in range(OFtest.shape[1]):
            test_u.append(((float)(OFtest[:,:,1][ridx,cidx])-math.pow(2,15))/64.0)
            test_v.append(((float)(OFtest[:,:,2][ridx,cidx])-math.pow(2,15))/64.0)
            gt_u.append(((float)(OFgt[:,:,1][ridx,cidx])-math.pow(2,15))/64.0)
            gt_v.append(((float)(OFgt[:,:,2][ridx,cidx])-math.pow(2,15))/64.0)
            gt_valid.append(OFgt[:,:,0][ridx,cidx])
    P=0
    for i in range(len(gt_valid)):
        if gt_valid[i]==1:
            SquareError = math.sqrt(math.pow((gt_u [i]-test_u [i]),2)+math.pow((gt_v [i]-test_v [i]),2))
            if SquareError>3.0:
                P=P+1
            SquareErrorList.append(SquareError)
    SquareErrorVector=np.asarray(SquareErrorList)
    MSEN = np.sum(SquareErrorVector)/len(SquareErrorVector)
    PEPN = float(P)/len(SquareErrorVector)
    return MSEN,PEPN
