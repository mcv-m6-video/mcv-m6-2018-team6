#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 18:45:48 2018

@author: ferran
"""
import numpy as np
import cv2
import sys
import os
import math
import matplotlib.pyplot as plt


def compute_error(block1, block2):
    return sum(sum(abs(block1-block2)**2))

def block_search(region_to_explore, block_to_search,block_size,area_size):

    x_size = region_to_explore.shape[0]
    y_size = region_to_explore.shape[1]

    min_diff = sys.float_info.max

    for row in range(x_size-block_size-1):
        for column in range(y_size-block_size-1):
            block2analyse = region_to_explore[row:row+block_size, column:column+block_size]
            diff = compute_error(block2analyse, block_to_search)
            if diff < min_diff:
                min_diff = diff
                x_mot = - row + area_size
                y_mot = column - area_size
    return x_mot, y_mot
    
def compute_block_matching(prev_img, curr_img,block_size,area_size):

    img2xplore = curr_img
    searchimg = prev_img

    x_blocks = img2xplore.shape[0]/block_size
    y_blocks = img2xplore.shape[1]/block_size

    #Add padding in the search image
    pad_searchimg = np.zeros([img2xplore.shape[0]+2*area_size,img2xplore.shape[1]+2*area_size])
    pad_searchimg[area_size:area_size+img2xplore.shape[0],area_size:area_size+img2xplore.shape[1]] = searchimg[:,:]

    motion_matrix = np.zeros([x_blocks, y_blocks, 2])
    
    for row in range(x_blocks):
        for column in range(y_blocks):
            # print "Computing block " + str(column)
            block_to_search = img2xplore[row*block_size:row*block_size+block_size, column*block_size:column*block_size+block_size]
#            region_to_explore = pad_searchimg[row*block_size:row*block_size+block_size+2*area_size, column*block_size:column*block_size+block_size+2*area_size]
            region_to_explore = pad_searchimg[row*block_size:row*block_size+block_size+2*area_size, column*block_size:column*block_size+block_size+2*area_size]
            x_mot, y_mot = block_search(region_to_explore, block_to_search,block_size,area_size)
            
            motion_matrix[row,column,0] = x_mot
            motion_matrix[row,column,1] = y_mot
            
    return motion_matrix

def create_compensated_image(prev_img, motion_matrix, block_size, x_blocks, y_blocks):
    x_size = prev_img.shape[0]
    y_size = prev_img.shape[1]

    comp_img = np.zeros([x_size, y_size])

    for x_pos in range(x_blocks):
        for y_pos in range(y_blocks):
#            comp_img[x_pos*block_size:x_pos*block_size+block_size,y_pos*block_size:y_pos*block_size+block_size]=prev_img[x_pos*block_size-motion_matrix[x_pos,y_pos,0]:x_pos*block_size+block_size-motion_matrix[x_pos,y_pos,0],y_pos*block_size+motion_matrix[x_pos,y_pos,1]:y_pos*block_size+block_size+motion_matrix[x_pos,y_pos,1]]
            comp_img[x_pos*block_size:x_pos*block_size+block_size,y_pos*block_size:y_pos*block_size+block_size]=prev_img[x_pos*block_size-int(motion_matrix[x_pos,y_pos,0]):x_pos*block_size+block_size-int(motion_matrix[x_pos,y_pos,0]),y_pos*block_size+int(motion_matrix[x_pos,y_pos,1]):y_pos*block_size+block_size+int(motion_matrix[x_pos,y_pos,1])]

    return comp_img

def create_OFtest(prev_img, motion_matrix, block_size, x_blocks, y_blocks):
    x_size = prev_img.shape[0]
    y_size = prev_img.shape[1]

    OFtest = np.zeros([x_size, y_size,2])
    for i in range(2):
        for x_pos in range(x_blocks):
            for y_pos in range(y_blocks):
    #            comp_img[x_pos*block_size:x_pos*block_size+block_size,y_pos*block_size:y_pos*block_size+block_size]=prev_img[x_pos*block_size-motion_matrix[x_pos,y_pos,0]:x_pos*block_size+block_size-motion_matrix[x_pos,y_pos,0],y_pos*block_size+motion_matrix[x_pos,y_pos,1]:y_pos*block_size+block_size+motion_matrix[x_pos,y_pos,1]]
                OFtest[x_pos*block_size:x_pos*block_size+block_size,y_pos*block_size:y_pos*block_size+block_size,i]=motion_matrix[x_pos,y_pos,i]

    return OFtest

def MSEN_PEPN(OFtest,OFgt):
    SquareErrorList=[]
    test_u=[]
    test_v=[]
    gt_u=[]
    gt_v=[]
    gt_valid=[]
    for ridx in range(OFtest.shape[0]):
        for cidx in range(OFtest.shape[1]):
            test_u.append(((float)(OFtest[:,:,0][ridx,cidx])))
            test_v.append(((float)(OFtest[:,:,1][ridx,cidx])))
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
    MSEN=0
    PEPN=0
    MSEN = np.sum(SquareErrorVector)/len(SquareErrorVector)
    PEPN = float(P)/len(SquareErrorVector)
    return MSEN,PEPN


def PlotOpticalFlow1(OFtest,OFgt):
    u_test_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_test_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_gt_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_gt_=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_testO=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    v_testO=np.zeros((OFtest.shape[0],OFtest.shape[1]))
    u_test=OFtest[:,:,0]
    v_test=OFtest[:,:,1]
    u_gt=OFgt[:,:,1]
    v_gt=OFgt[:,:,2]
#    print ('error')
    valid_gt=OFgt[:,:,0]
    for ridx in range(OFtest.shape[0]):
        for cidx in range(OFtest.shape[1]):
            u_test_ [ridx,cidx]=((float)(u_test [ridx,cidx]))
            v_test_ [ridx,cidx]=((float)(v_test [ridx,cidx]))
            u_gt_ [ridx,cidx]=((float)(u_gt [ridx,cidx])-math.pow(2,15))/64.0
            v_gt_ [ridx,cidx]=((float)(v_gt [ridx,cidx])-math.pow(2,15))/64.0
    for ri in range(valid_gt.shape[0]):
        for ci in range(valid_gt.shape[1]):
            if valid_gt [ri,ci] == 1:
                u_testO [ri,ci]= u_test_ [ri,ci] /100.0
                v_testO [ri,ci]= v_test_ [ri,ci] /100.0
    
    x, y = np.meshgrid(np.arange(0, OFtest.shape[1], 1), np.arange(0, OFtest.shape[0], 1))
    M=np.hypot(u_testO, v_testO)
    Q=plt.quiver(x, y, u_testO, v_testO, M, units='x', pivot='tip', width=0.022,
               scale=1 / 0.15)
    plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
    plt.scatter(x, y, color='k', s=5)
    plt.show()
    

