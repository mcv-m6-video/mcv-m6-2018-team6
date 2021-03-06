#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:41:34 2018

@author: yixiao
"""
#--------------------------------------------------------------------
# M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring
# Team06: YI XIAO, Juan Felipe Montesinos,Ferran Carrasquer

# week 1: Introduction, Databases, Evaluation Metrics

# This is the script to be launched for week 1 tasks
#--------------------------------------------------------------------

import evaluation as ev

# the folder path of test A, B and the groundtruth
# the folder path of optical flow results and ground truth
folder_pathA='/home/yixiao/Documents/M6_project/datasets/testAB/testA'
folder_pathB='/home/yixiao/Documents/M6_project/datasets/testAB/testB'
gt_path='/home/yixiao/Documents/M6_project/datasets/highway_GT'
OFgt_path='/home/yixiao/Documents/M6_project/datasets/opticalflow_GT'
OFtest_path='/home/yixiao/Documents/M6_project/datasets/opticalflow_test'

# task1.1: evaluation for all frames with metrics: precision, recall and f1 score 
precision_A, recall_A, f1_score_A = ev.evaluateAllFrames(folder_pathA, gt_path)
precision_B, recall_B, f1_score_B = ev.evaluateAllFrames(folder_pathB, gt_path)


# task2: Temporal Analysis of the Results
# task2.1 True Positives and Total Foreground vs time 
# testA
ev.temproalTP(folder_pathA,gt_path)
## testB
ev.temproalTP(folder_pathB,gt_path)

## task2.2 F1 score vs time 
# testA
ev.temproalFscore(folder_pathA,gt_path)
# testB
ev.temproalFscore(folder_pathB,gt_path)

## task3: Quantitative evaluation of optical flow
# seq 45
OFtest = cv2.imread(os.path.join(OFtest_path, 'LKflow_000045_10.png'), -1)
OFgt = cv2.imread(os.path.join(OFgt_path, '000045-10.png'), -1)
MSEN1,PEPN1 = ev.MSEN_PEPN(OFtest,OFgt)
# seq 157
OFtest = cv2.imread(os.path.join(OFtest_path, 'LKflow_000157_10.png'),-1)
OFgt = cv2.imread(os.path.join(OFgt_path, '000157-10.png'),-1)
MSEN2,PEPN2 = ev.MSEN_PEPN(OFtest,OFgt)
