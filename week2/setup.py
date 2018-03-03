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
import glob
from sklearn.metrics import precision_recall_fscore_support as PRFmetrics

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

class Original: 
    def __init__(self,name,im_dir,gt_dir,color=False):
        self.color = color
        self.name = name
        self.im_dir = im_dir
        self.gt_dir = gt_dir     
        
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
        
#Defining a class to perform gaussian-based motion estimation
class gaussian1D(Original):
    mean = None
    std  = None

        
    def get_1D(self,frame_list):
        im_patch = []
        #Stacking all the frames in a single 3D/4D array (depending if convert to
        #grayscale or not)
        for i in sorted(frame_list):
            im_dir = os.path.join(self.im_dir, i)
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
        diff = np.abs(self.mean-im)
        foreground = (diff >= th*(self.std+2))
        foreground = foreground.astype(int)
        return foreground

    def evaluateSeveralFrames(self,frame_list):
        predVector = []
        trueVector = []
        for i in sorted(frame_list):
            im_dir = os.path.join(self.im_dir, i)
            image = cv2.imread(im_dir,0)
            foreground = self.get_motion(image,2)
            for ridx in range(image.shape[0]):
                for cidx in range(image.shape[1]):
                    predVector.append(foreground[ridx,cidx])
        for filename in sorted(glob.glob(os.path.join(self.gt_dir, 'gt00*.png'))):
            gtImage = cv2.imread(filename,0)
            for ridx in range(gtImage.shape[0]):
                for cidx in range(gtImage.shape[1]):
                    trueVector.append(gtImage[ridx,cidx])

        trueArray = np.asarray(trueVector)
        predArray = np.asarray(predVector)        
        for j in range(len(trueArray)):
            if trueArray[j]==255:
                trueArray[j] = 1
            else:
                trueArray[j] = 0
        precision, recall,f1_score,support = PRFmetrics(trueArray, predArray, average='binary')
        return precision, recall, f1_score
    


