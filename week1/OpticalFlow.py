#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:46:47 2018

@author: yixiao

#--------------------------------------------------------------------
# M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring
# Team06: YI XIAO, Juan Felipe Montesinos,Ferran Carrasquer

# week 1: Introduction, Databases, Evaluation Metrics

# This .py file is for defining some functions of Optical Flow evaluation
#--------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation 

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
    return MSEN,PEPN,SquareErrorVector

def histSquareError(data):
    cm = plt.cm.get_cmap('RdYlBu_r')
    n, bins, patches = plt.hist(data,25, normed=1)  
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.title("Histogram of square error")
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.show()

def visulizeError(OFtest,OFgt):
    errorMap=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_test_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_test_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_gt_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_gt_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_test=OFtest[:,:,1]
    v_test=OFtest[:,:,2]
    u_gt=OFgt[:,:,1]
    v_gt=OFgt[:,:,2]
    valid_gt=OFgt[:,:,0]
    for ridx in range(OFtest.shape[0]):
        for cidx in range(OFtest.shape[1]):
            u_test_ [ridx,cidx]=((float)(u_test [ridx,cidx])-math.pow(2,15))/64.0
            v_test_ [ridx,cidx]=((float)(v_test [ridx,cidx])-math.pow(2,15))/64.0
            u_gt_ [ridx,cidx]=((float)(u_gt [ridx,cidx])-math.pow(2,15))/64.0
            v_gt_ [ridx,cidx]=((float)(v_gt [ridx,cidx])-math.pow(2,15))/64.0
    for ri in range(valid_gt.shape[0]):
        for ci in range(valid_gt.shape[1]):
            if valid_gt [ri,ci] == 1:
                SquareError = math.sqrt(math.pow((u_gt_ [ri,ci]-u_test_ [ri,ci]),2)+math.pow((v_gt_ [ri,ci]-v_test_ [ri,ci]),2))
                errorMap[ri,ci]=SquareError
    plt.imshow(errorMap)
    plt.colorbar()
    plt.show()
    

def PlotOpticalFlow(OFtest,OFgt):
    u_test_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_test_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_gt_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_gt_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_testO=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_testO=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_test=OFtest[:,:,1]
    v_test=OFtest[:,:,2]
    u_gt=OFgt[:,:,1]
    v_gt=OFgt[:,:,2]
    valid_gt=OFgt[:,:,0]
    for ridx in range(OFtest.shape[0]):
        for cidx in range(OFtest.shape[1]):
            u_test_ [ridx,cidx]=((float)(u_test [ridx,cidx])-math.pow(2,15))/64.0
            v_test_ [ridx,cidx]=((float)(v_test [ridx,cidx])-math.pow(2,15))/64.0
            u_gt_ [ridx,cidx]=((float)(u_gt [ridx,cidx])-math.pow(2,15))/64.0
            v_gt_ [ridx,cidx]=((float)(v_gt [ridx,cidx])-math.pow(2,15))/64.0
    for ri in range(valid_gt.shape[0]):
        for ci in range(valid_gt.shape[1]):
            if valid_gt [ri,ci] == 1:
                u_testO [ri,ci]= u_test_ [ri,ci] /200.0
                v_testO [ri,ci]= v_test_ [ri,ci] /200.0
    
    x, y = np.meshgrid(np.arange(0, OFtest.shape[1], 1), np.arange(0, OFtest.shape[0], 1))
    plt.quiver(x, y, u_testO, v_testO, scale=1, hatch=' ', alpha = 0.3, linewidth = 0.001)
    plt.show()
