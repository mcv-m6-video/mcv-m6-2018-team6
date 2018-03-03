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
import os
import numpy as np
import cv2


#Defining a class to perform gaussian-based motion estimation
class gaussian1D:
    mean = None
    std  = None 
    def __init__(self, color):
        self.color = color
        
    def get_1D(self,frame_list,frame_dir):
        im_patch = []
        #Stacking all the frames in a single 3D/4D array (depending if convert to
        #grayscale or not)
        for i in sorted(frame_list):
            im_dir = os.path.join(frame_dir, i)
            if self.color==False:
                image = cv2.cvtColor(cv2.imread(im_dir,-1),cv2.COLOR_BGR2GRAY)
            else:
                image = cv2.imread(im_dir,-1)
            im_patch.append(image)
            
        im_patch = np.asarray(im_patch)
        if self.color==True:
            self.mean = im_patch.mean(axis=3)
            self.std  = im_patch.std(axis=3)
        else:
            self.mean = im_patch.mean(axis=0)
            self.std  = im_patch.std(axis=0)

    def get_motion(self,im,th):
        if self.color == False:
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im = np.asarray(im)
        print 'im: '+str(im.shape)
        print 'self: '+str(self.mean.shape)
        diff = np.abs(self.mean-im)
        foreground = (diff >= th*(self.std+2))
        foreground = foreground.astype(int)
        return foreground
