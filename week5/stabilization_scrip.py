#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:19:21 2018

@author: yixiao
"""

from __future__ import division
import datetime
import imageio
import numpy as np
import cv2
import os
import stabilization


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)

    
"""
============================== this is the traffic dataset ===========================
"""
data_dir = '/home/yixiao/Documents/M6_project/week5/images/'

frames_train = []


print 'Reading background modeling files:...'
for i in range(400,510):
    frames_train.append('in000'+str(i)+'.jpg')
    print 'in000'+str(i)+'.jpg loaded'

im_curr = os.path.join(data_dir, frames_train[0])
curr_img = cv2.imread(im_curr,-1)
curr_img = cv2.cvtColor(curr_img,cv2.COLOR_BGR2GRAY) 
results_list_dir=[]
trans_img=curr_img
n=0
for i in sorted(frames_train):

    curr_img=trans_img  
    im_future = os.path.join(data_dir, frames_train[n])
    future_img = cv2.imread(im_future,-1)
    future_img = cv2.cvtColor(future_img,cv2.COLOR_BGR2GRAY)
    print im_future
    trans_img= stabilization.image_translate(curr_img,future_img)
    cv2.imwrite('stabilization_trafficUAB/000'+str(i)+'.png',trans_img)
    #results_list_dir.append('results_trafficuab/0000'+str(i)+'.png')
    n=n+1 
    
#create_gif(results_list_dir, 0.1)
#create_gif(frames_train, 0.1)