#title           :main.py
#description     :MSc in Computer Vision
#author          :Juan Felipe Montesinos
#date            :14-march-2018
#version         :0.3
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

from sklearn.metrics import precision_recall_curve
import scipy.stats as ndis
import math
from skimage import morphology
from sklearn.metrics import auc


# Input the pair of image and gt, this function will output the TP, FP, TN, FN
def index_rep(a,b,c):
    no_c = ~c 
    no_c = no_c.astype(int)
    c = c.astype(int)
    out = np.multiply(a,no_c) + np.multiply(b,c)
    return out
#def Projection(bg,fg):
#    #dot_product = np.dot(bg,fg)
#    a,b,c=bg.shape
#    norm_bg = np.linalg.norm(bg,axis=2)
#    bg_hat=np.zeros(bg.shape)
#    for i in range(a):
#        for j in range(b):
#            bg_hat[i,j,:] = bg[i,j,:]/norm_bg[i,j].astype(float)
#    BD=np.zeros(bg.shape)
#    for i in range(a):
#        for j in range(b):
#            BD[i,j,:] = np.dot(fg[i,j,:],bg_hat[i,j,:])*bg_hat[i,j,:]
#    CD = fg-BD
#    return CD,BD
def Projection(bg,fg):
    #dot_product = np.dot(bg,fg)
    a,b,c=bg.shape
    norm_bg = np.linalg.norm(bg,axis=2)
    bg_hat=np.zeros(bg.shape)
    for i in range(a):
        for j in range(b):
            bg_hat[i,j,:] = bg[i,j,:]/norm_bg[i,j].astype(float)
    BD=np.zeros(bg.shape)
    BD_norm=np.zeros(norm_bg.shape)
    for i in range(a):
        for j in range(b):
            BD_norm[i,j] = np.dot(fg[i,j,:],bg_hat[i,j,:])
            BD[i,j,:] = BD_norm[i,j]*bg_hat[i,j,:]
    CD = fg-BD
    BD=norm_bg/BD_norm
    return CD,BD
class Original:
    def __init__(self,name,im_dir,gt_dir,color='gray',shadow_removal = True):
        self.color = color
        self.name = name
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.frames_train = []
        self.gt_train = []
        self.shadows = shadow_removal
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
        plt.close()
    def evaluateOneFrame(frame,gt):
        predVector = []
        trueVector = []
        frame_flat = np.array(frame).flatten()
        gt_flat = np.array(gt).flatten()
        i_g=0
        for i in gt_flat:
            if i==255 or i==170:
                trueVector.append(1)
                predVector.append(frame_flat[i_g])
            else:
                trueVector.append(0)
                predVector.append(frame_flat[i_g])
            i_g = i_g+1
            
        trueArray = np.asarray(trueVector)
        predArray = np.asarray(predVector)
        _, _,f1_score,_ = PRFmetrics(trueArray, predArray,average='binary')
        precision, recall,_ = precision_recall_curve(trueArray, predArray, average='binary')
        AUC= auc(recall, precision)
#        TP=0
#        TN=0
#        FP=0
#        FN=0
#        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
#        TP = np.sum(np.logical_and(predArray == 1, trueArray == 1))
#        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
#        TN = np.sum(np.logical_and(predArray == 0, trueArray == 0))
#        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
#        FP = np.sum(np.logical_and(predArray == 1, trueArray == 0))
#        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
#        FN = np.sum(np.logical_and(predArray == 0, trueArray == 1))
#        # for the gt, we only consider two classes(0,255) represent background and motion respectively.    
        return precision, recall, f1_score, AUC                 

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
    def Read(self,dataset):
        if dataset == 'highway':
            self.frames_train = []
            self.gt_train = []
            self.frames_test = []
            self.gt_test = []
            for i in range(1050,1201):
                self.frames_train.append('in00'+str(i)+'.jpg')
                self.gt_train.append('gt00'+str(i)+'.png')
                print 'in00'+str(i)+'.jpg loaded'
            for i in range(1201,1351):
                self.frames_test.append('in00'+str(i)+'.jpg')
                self.gt_test.append('gt00'+str(i)+'.png')
        elif dataset == 'fall':
            self.frames_train = []
            self.gt_train = []
            self.frames_test = []
            self.gt_test = []
            for i in range(1460,1511):
                self.frames_train.append('in00'+str(i)+'.jpg')
                self.gt_train.append('gt00'+str(i)+'.png')
            for i in range(1511,1561):
                self.frames_test.append('in00'+str(i)+'.jpg')
                self.gt_test.append('gt00'+str(i)+'.png')
        elif dataset == 'traffic':
            self.frames_train = []
            self.gt_train = []
            self.frames_test = []
            self.gt_test = []
            for i in range(950,1001):
                if i >=1000:
                    self.frames_train.append('in00'+str(i)+'.jpg')
                    self.gt_train.append('gt00'+str(i)+'.png')
                else:
                    self.frames_train.append('in000'+str(i)+'.jpg')
                    self.gt_train.append('gt000'+str(i)+'.png')
            for i in range(1001,1051):
                self.frames_test.append('in00'+str(i)+'.jpg')
                self.gt_test.append('gt00'+str(i)+'.png')           
                            
#Defining a class to perform gaussian-based motion estimation
class gaussian1D(Original):
    mean = None
    std  = None

        
    def get_1D(self,frame_list):
        im_patch = []
        spectral_patch = []
        #Stacking all the frames in a single 3D/4D array (depending if convert to
        #grayscale or not)
        for i in sorted(frame_list):
            im_dir = os.path.join(self.im_dir, i)
            if self.color=='gray':
                image = cv2.cvtColor(cv2.imread(im_dir,-1),cv2.COLOR_BGR2GRAY)
                im_patch.append(image)
            elif self.color=='RGB':
                image = cv2.imread(im_dir,-1)
                im_patch.append(image)
            elif self.color=='HSV':
                image = cv2.cvtColor(cv2.imread(im_dir,-1),cv2.COLOR_BGR2HSV)
                im_patch.append(image)
                channelH = image[:,:,0] 
                channelV = image[:,:,2]
                spectral_ratio = channelH/channelV.astype(float)
                spectral_patch.append(spectral_ratio)
        spectral_patch=np.asarray(spectral_patch)
        im_patch = np.asarray(im_patch)
        if self.color=='gray':
            self.mean = im_patch.mean(axis=0)
            self.std  = im_patch.std(axis=0)
        elif self.color=='RGB':
            self.mean = im_patch.mean(axis=0)
            self.std  = im_patch.std(axis=0)
        elif self.color=='HSV' :
            self.mean = im_patch.mean(axis=0)
            self.std  = im_patch.std(axis=0)
            self.mean_s = spectral_patch.mean(axis=0)
            self.std_s  = spectral_patch.std(axis=0)
    
    def get_motion(self,im,th,shadow_th = 0.5):
        if self.color == 'gray':
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            im = np.asarray(im)
            diff = self.mean - im
            foreground = (np.abs(diff) >= th*(self.std+2))        
            if self.shadows == True:
                shadows_criteria = diff/(self.std+1)
                shadows = shadows_criteria > shadow_th
                shadows = shadows & foreground
                foreground = foreground & ~shadows
                shadows = shadows.astype(int)
            else:
                shadows = None
            foreground = foreground.astype(int)
        elif self.color == 'RGB':
            im = np.asarray(im)
            channelR = im[:,:,2] 
            channelG = im[:,:,1]
            channelB = im[:,:,0]
            diffR = np.abs(self.mean[:,:,2]-channelR)
            diffG = np.abs(self.mean[:,:,1]-channelG)
            diffB = np.abs(self.mean[:,:,0]-channelB)
            foreground_R = (diffR >= th*(self.std[:,:,2]+2))
            foreground_G = (diffG >= th*(self.std[:,:,1]+2))
            foreground_B = (diffB >= th*(self.std[:,:,0]+2))
            foreground = np.logical_and(foreground_R,foreground_G,foreground_B)
            if self.shadows == True:
                CD,BD = Projection(self.mean,im)
                CD_cond = np.linalg.norm(CD,axis=2)<10
                np.savetxt('CD_cond.txt',np.linalg.norm(CD,axis=2))
                np.savetxt('BD.txt',BD)
                w=BD>0.5
                w2 = BD<1
                m = np.logical_and(w,CD_cond)
                shadows = np.logical_and(w2,m)
                shadows = shadows & foreground
                foreground = foreground & ~shadows
                shadows = shadows.astype(int)
            else:
                shadows = None
            foreground = foreground.astype(int)
        elif self.color == 'HSV':
            im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            channelH = im[:,:,0] 
            channelS = im[:,:,1]
            #channelV = im[:,:,2]
            diffH = np.abs(self.mean[:,:,0]-channelH)
            diffS = np.abs(self.mean[:,:,1]-channelS)
            #diffV = np.abs(self.mean[:,:,2]-channelV)
            foreground_H = (diffH >= th*(self.std[:,:,0]+2))
            foreground_S = (diffS >= th*(self.std[:,:,1]+2))
            #foreground_V = (diffV >= th*(self.std[:,:,2]+2))  
            # we dont take into account the V channel 
            foreground = np.logical_and(foreground_H,foreground_S)
            if self.shadows == True:
                channelV = im[:,:,2]
                spectral_ratio = channelH/channelV.astype(float)
                shadows_criteria = (spectral_ratio-self.mean_s)/(self.std_s)
                shadows = shadows_criteria > shadow_th
                shadows = shadows & foreground
                foreground = foreground & ~shadows
                shadows = shadows.astype(int)   
            else:
                shadows = None
            foreground = foreground.astype(int)
        return foreground, shadows

#    def evaluateSeveralFrames(self,frame_list,gt_list,th):
#        predArray = []
#        trueVector = []
#        n = 0 
#        for i in sorted(gt_list):
#            im_dir = os.path.join(self.im_dir, frame_list[n])
#            image = cv2.imread(im_dir,-1)
#            foreground = self.get_motion(image,th)
#            gt_dir = os.path.join(self.gt_dir, i)
#            gtImage = cv2.imread(gt_dir,0)
#            foreground_flat = np.array(foreground).flatten() 
#            gtImage_flat = np.array(gtImage).flatten()
#            motionVector = gtImage_flat == 255
#            bgVector = gtImage_flat ==0 | gtImage_flat == 50
#            trueVector_fg = foreground_flat[motionVector==True]
#            trueVector_fg = trueVector_fg.astype(int)
#            trueVector_bg = foreground_flat[bgVector==True]
#            trueVector_bg = trueVector_bg.astype(int)
#            motionVector = motionVector.astype(int)
#            bgVector = bgVector.astype(int)
#            trueVector=np.append(trueVector_fg,trueVector_bg)
#            predVector =np.append(motionVector,bgVector)
#            trueArray = np.append(trueArray,trueVector)
#            predVector = np.append(predArray,predVector)
#            n=n+1
#        precision, recall,f1_score,support = PRFmetrics(trueArray, predArray,average='binary')
#        return precision, recall, f1_score
    def evaluateSeveralFrames(self,frame_list,gt_list,th,shadow_th=0.5):
        predVector = []
        trueVector = []
        n = 0 
        for i in sorted(gt_list):
            im_dir = os.path.join(self.im_dir, frame_list[n])
            image = cv2.imread(im_dir,-1)
            foreground = self.get_motion(image,th,shadow_th)
            gt_dir = os.path.join(self.gt_dir, i)
            gtImage = cv2.imread(gt_dir,0)
            foreground_flat = np.array(foreground).flatten() 
            gtImage_flat = np.array(gtImage).flatten()
            i_g=0
            for i in gtImage_flat:
                if i==255:
                    trueVector.append(1)
                    predVector.append(foreground_flat[i_g])
                elif i==0 or i==50:
                    trueVector.append(0)
                    predVector.append(foreground_flat[i_g])
                i_g = i_g+1
            n = n+1
        trueArray = np.asarray(trueVector)
        predArray = np.asarray(predVector)        
        precision, recall,f1_score,support = PRFmetrics(trueArray, predArray,average='binary')
        return precision, recall, f1_score
    
    
    def AUCvsP(self,frames_list,gt_list, th, P_range):
        AUC_list = []
        for P in P_range:
            print "Filtering objects smaller than " + str(P)
            n=0
            AUC_p = []
            for i in sorted(gt_list):
                im_dir = os.path.join(self.im_dir, frames_list[n])
                image = cv2.imread(im_dir,-1)
                foreground = self.get_motion(image,th)
                gt_dir = os.path.join(self.gt_dir, i)
                gtImage = cv2.imread(gt_dir,0)
                AFimage = morphology.remove_small_objects(foreground.astype(bool), min_size=P)
                AFimage = AFimage.astype(int)
                _, _, _, PRauc_perImg = self.evaluateOneFrame(AFimage, gtImage)
                AUC_p.append(PRauc_perImg)
                n=n+1
            AUC_Array = np.asarray(AUC_p)
            mean_AUC=np.mean(AUC_Array)
            AUC_list.append(mean_AUC)
        AUC_allP=np.asarray(AUC_list)
        return AUC_allP

    def allvsalpha(self,frame_list,gt_list,th_vector,shadows_vector=[0.5]):
        self.F1_vector = []
        self.precision_vector = []
        self.recall_vector = []
        self.x = th_vector
        if self.shadows==True:
            self.shadowth=shadows_vector
        for i in th_vector:
            for j in shadows_vector:
                precision, recall, F1 = self.evaluateSeveralFrames(frame_list,gt_list,i,j)
                self.F1_vector.append(F1)
                self.precision_vector.append(precision)
                self.recall_vector.append(recall)
                if self.shadows==False:
                    print str(i*100/self.x.max())+'% completed'
                else:
                    print str(j*100/self.shadowth.max())+'% completed'
            
    def saveAllvsalpha(self):
        np.savetxt(self.name+'_F1.txt',self.F1_vector)
        np.savetxt(self.name+'_precision.txt',self.precision_vector)
        np.savetxt(self.name+'_recall.txt',self.recall_vector)
        np.savetxt(self.name+'_x.txt',self.x)
        if self.shadows==True:
            np.savetxt(self.name+'_shadowth.txt',self.shadowth)
    def LoadAllvsalpha(self):
        self.F1_vector= np.loadtxt(self.name+'_F1.txt')
        self.precision_vector = np.loadtxt(self.name+'_precision.txt')
        self.recall_vector = np.loadtxt(self.name+'_recall.txt')
        self.x = np.loadtxt(self.name+'_x.txt')
        if self.shadows==True:
            self.shadowth == np.loadtxt(self.name+'_shadowth.txt')
    def PlotMeanStd(self):
        #PLOTTING MEAN AND STD AS IMAGES AND STORING EM
        if self.color =='gray':
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
                plt.savefig(self.name+'_mean_stdplot''.png', bbox_inches='tight', pad_inches = 0)
                plt.close()
        else:
            for j in range(self.mean.ndim):
                fig, axes = plt.subplots(nrows=2, ncols=1)
                im = axes[0].imshow(self.mean[:,:,j].astype(int), vmin=0, vmax=255)
                axes[1].imshow(self.std[:,:,j].astype(int), vmin=0, vmax=255)
                cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
                plt.colorbar(im, cax=cax, **kw)
                for i in range(len(axes)):
                    axes[i].axes.get_xaxis().set_visible(False)
                    axes[i].axes.get_yaxis().set_visible(False)
                axes[0].set_title('Mean values')
                axes[1].set_title('Std values')
                plt.savefig(self.name+'_mean_stdplot'+str(j)+'.png', bbox_inches='tight', pad_inches = 0)
                plt.close()
                
class MOG(gaussian1D):
        def get_1D(self,frame_list):
            cv2.backgroundSubtractor = cv2.BackgroundSubtractorMOG(history=100, 
                        nmixtures=5, backgroundRatio=0.7, noiseSigma=0)
            for i in sorted(frame_list):
                im_dir = os.path.join(self.im_dir, i)
                image = cv2.imread(im_dir,-1)
                print "Opening background"
                cv2.backgroundSubtractor.apply(image, learningRate=0.5)
        def get_motion(self,image,th=0.5):
            print "Opening background", image
            foreground = cv2.backgroundSubtractor.apply(image, learningRate=0)
            foreground =np.asarray(foreground)
            foreground = foreground.astype(bool)
            foreground = foreground.astype(int)
            return foreground
def bgr2gray(rgb):

    b,g,r = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray          
class adaptative(gaussian1D):
        def get_motion(self,im,th):
            if self.color == 'gray':
                #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                im = bgr2gray(im)
                im = np.asarray(im)
                diff = np.abs(self.mean-im)
                foreground = (diff >= th[1]*(self.std+2))
                index_rep(self.mean,th[0]*im+(1-th[0])*self.mean,foreground == False)
                index_rep(self.std,np.sqrt(th[0]*np.multiply((im-self.mean),(im-self.mean))+(1-th[0])*np.multiply(self.std,self.std)),foreground == False)
                foreground = foreground.astype(int)
                return foreground
        def allvsalpha(self,frame_list,gt_list,alpha,beta):
            self.F1_vector = np.zeros(shape=[len(alpha),len(beta)])
            self.precision_vector = np.zeros(shape=[len(alpha),len(beta)])
            self.recall_vector = np.zeros(shape=[len(alpha),len(beta)])
            self.alpha = alpha
            self.beta = beta
            i_a = 0
            i_b = 0
            for i in alpha:
                i_b = 0
                for j in beta:
                    precision, recall, F1 = self.evaluateSeveralFrames(frame_list,gt_list,[j,i])
                    self.F1_vector[i_a,i_b] = F1
                    self.precision_vector[i_a,i_b] = precision
                    self.recall_vector[i_a,i_b] = recall
                    i_b = i_b +1
                    print str(i*100/self.alpha.max())+'% completed'
                i_a = i_a +1
        def saveAllvsalpha(self):
            np.savetxt(self.name+'_F1.txt',self.F1_vector)
            np.savetxt(self.name+'_precision.txt',self.precision_vector)
            np.savetxt(self.name+'_recall.txt',self.recall_vector)
            np.savetxt(self.name+'_alpha.txt',self.alpha)
            np.savetxt(self.name+'_beta.txt',self.beta)
    
        def LoadAllvsalpha(self):
            self.F1_vector= np.loadtxt(self.name+'_F1.txt')
            self.precision_vector = np.loadtxt(self.name+'_precision.txt')
            self.recall_vector = np.loadtxt(self.name+'_recall.txt')
            self.alpha = np.loadtxt(self.name+'_alpha.txt')
            self.beta = np.loadtxt(self.name+'_beta.txt')
