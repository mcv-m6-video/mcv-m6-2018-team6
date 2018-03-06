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
import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    def animacion(self,frame_list):
        fig = plt.figure()
        plt.axis('off')
        
        ims = []
        for i in sorted(frame_list):
            im_dir = os.path.join(self.im_dir, i)
            print i+' ...loaded!'
            im = mgimg.imread(im_dir)
            imgplot = plt.imshow(im)
            ims.append([imgplot])
        
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)      
        ani.save(self.name+'.gif')
        
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

    def errorPainting(self,frame_list,gt_list,results_list_dir):
        for i in range(len(frame_list)):
            im_dir = os.path.join(self.im_dir, frame_list[i])
            gt_dir = os.path.join(self.gt_dir, gt_list[i])
            res = cv2.imread(results_list_dir[i],0)
            im = cv2.imread(im_dir,-1)
            gt = cv2.imread(gt_dir,0)
            gt[gt==255]=1
            gt[gt==50]=0
            gt[gt==85]=0
            gt[gt==170]=0
            res=res.astype(bool)
            gt=gt.astype(bool)
            error1 = (res & ~gt) 
            error2 = (~res & gt)
            error = error1 | error2
            b = im[:,:,0]
            g = im[:,:,1]
            r = im[:,:,2]
            b[error==True]=0
            g[error==True]=0
            g[error1==True]=255
            r[error2==True]=255
            im[:,:,0]=b
            im[:,:,1]=g
            im[:,:,2]=r
            cv2.imwrite(results_list_dir[i],im)
            
                        

                
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

    def evaluateSeveralFrames(self,frame_list,gt_list,th):
        predVector = []
        trueVector = []
        for i in sorted(frame_list):
            im_dir = os.path.join(self.im_dir, i)
            image = cv2.imread(im_dir,-1)
            foreground = self.get_motion(image,th)
        for i in sorted(gt_list):
            gt_dir = os.path.join(self.gt_dir, i)
            gtImage = cv2.imread(gt_dir,0)
            for ridx in range(gtImage.shape[0]):
                for cidx in range(gtImage.shape[1]):
                    if gtImage [ridx,cidx]==255:
                        trueVector.append(1)
                        predVector.append(foreground[ridx,cidx])
                    elif gtImage [ridx,cidx]==0 or gtImage [ridx,cidx]==50:
                        trueVector.append(0) 
                        predVector.append(foreground[ridx,cidx])
        
        trueArray = np.asarray(trueVector)
        predArray = np.asarray(predVector)        
        precision, recall,f1_score,support = PRFmetrics(trueArray, predArray,average='binary')
        return precision, recall, f1_score
    

    def allvsalpha(self,frame_list,gt_list,th_vector):
        self.F1_vector = []
        self.precision_vector = []
        self.recall_vector = []
        self.x = th_vector
        for i in th_vector:
            precision, recall, F1 = self.evaluateSeveralFrames(frame_list,gt_list,i)
            self.F1_vector.append(F1)
            self.precision_vector.append(precision)
            self.recall_vector.append(recall)
            print str(i*10)+'% completed'
            
    def saveAllvsalpha(self):
        np.savetxt(self.name+'_F1.txt',self.F1_vector)
        np.savetxt(self.name+'_precision.txt',self.precision_vector)
        np.savetxt(self.name+'_recall.txt',self.recall_vector)
        np.savetxt(self.name+'_x.txt',self.x)
    
    def LoadAllvsalpha(self):
        self.F1_vector= np.loadtxt(self.name+'_F1.txt')
        self.precision_vector = np.loadtxt(self.name+'_precision.txt')
        self.recall_vector = np.loadtxt(self.name+'_recall.txt')
        self.x = np.loadtxt(self.name+'_x.txt')

    def PlotMeanStd(self):
        #PLOTTING MEAN AND STD AS IMAGES AND STORING EM
        fig, axes = plt.subplots(nrows=2, ncols=1)
        im = axes[0].imshow(self.mean.astype(int), vmin=0, vmax=255)
        axes[1].imshow(self.std.astype(int), vmin=0, vmax=255)
        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        for i in range(len(axes)):
            axes[i].axes.get_xaxis().set_visible(False)
            axes[i].axes.get_yaxis().set_visible(False)
        axes[0].set_title('Mean values')
        axes[1].set_title('Std values')
        plt.savefig(self.name+'_mean_stdplot.png', bbox_inches='tight', pad_inches = 0)
    
        
