from skimage import morphology
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support as PRFmetrics
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import binary_fill_holes, generate_binary_structure
import math
#import main2

print 'MSc in Computer Vision Barcelona'
print 'Universidad Politecnica de Catalunya'
print '......................................'
print '......................................'



diamond3= np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
diamond5= np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]], np.uint8)
diamond7= np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]], np.uint8)
diagonal45= np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], np.uint8)
diagonal135= np.array([[0,0,0,0,0,0,1],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],[0,1,0,0,0,0,0],[1,0,0,0,0,0,0]], np.uint8)
rectangle3=np.ones((3,3),np.uint8)
rectangle5=np.ones((5,5),np.uint8)
rectangle7=np.ones((7,7),np.uint8)
rectangle10=np.ones((10,10),np.uint8)
rectangle15=np.ones((15,15),np.uint8)
rectangle20=np.ones((20,20),np.uint8)
rectangle30=np.ones((30,30),np.uint8)
rectangle40=np.ones((40,40),np.uint8)
rectangle50=np.ones((50,50),np.uint8)
line10=np.ones((2,10),np.uint8)
line20=np.ones((4,20),np.uint8)
line30=np.ones((6,30),np.uint8)
line2=np.ones((10,2),np.uint8)
line4=np.ones((150,20),np.uint8)
line6=np.ones((10,60),np.uint8)
#"""

"""
============================== this is the traffic dataset ===========================
"""
data_dir = '/home/ferran/Desktop/M6_project/datasets/traffic/input'
gt_dir   = '/home/ferran/Desktop/M6_project/datasets/traffic/groundtruth'
###Creating a list of frame names to perform a background model
frames_train = []
gt_train = []

frames_test = []
gt_test = []

print 'Reading background modeling files:...'
for i in range(950,1000):
    frames_train.append('in000'+str(i)+'.jpg')
    gt_train.append('gt000'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg loaded'
for i in range(1000,1051):
    frames_test.append('in00'+str(i)+'.jpg')
    gt_test.append('gt00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg loaded'
print 'Done!'
    
im_patch=[]
for i in sorted(frames_train):
    im_dir = os.path.join(data_dir, i)
    image = cv2.cvtColor(cv2.imread(im_dir,-1),cv2.COLOR_BGR2GRAY)
    im_patch.append(image)
    im_Array = np.asarray(im_patch)
    mean = im_Array.mean(axis=0)
    std  = im_Array.std(axis=0)

el = generate_binary_structure(2,1)

threshold = np.arange(0,10,1)
P_range = np.arange(0,1100,100)
AUC_allP = []

P = 1000
print "Filtering objects smaller than " + str(P)
precision_list = []
recall_list = [] 
for th in threshold:
    n=0
    print n
    predVector = []
    trueVector = []
    print "using threshold equal to " + str(th)
    for i in sorted(gt_test):
        im_d = os.path.join(data_dir, frames_test[n])
        gt_d = os.path.join(gt_dir, i)
        im = cv2.imread(im_d,-1)
        gt = cv2.imread(gt_d,0)
#        image_translate(curr_img,future_img)####
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im = np.asarray(im)
#        image_translate(curr_img,future_img)####
        diff = np.abs(mean-im)
        foreground = (diff >= th*(std+2))
        foreground = foreground.astype(int)
        HFimage = binary_fill_holes(foreground, el)
        HFimage = HFimage.astype(int)
        AFimage = morphology.remove_small_objects(HFimage.astype(bool), min_size=P)
#        AFimage=AFimage.astype(int)
        
        AFimage = np.array(AFimage,dtype=np.uint8)
        Mop_image= cv2.morphologyEx(AFimage, cv2.MORPH_OPEN,diagonal135 )
#        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        AFimage1 = np.array(Mop_image,dtype=np.uint8)
        Mop_image= cv2.morphologyEx(AFimage1, cv2.MORPH_OPEN,rectangle10)
        Mop_image=Mop_image.astype(int)
#        if th==1:
#            plt.imshow(gt, cmap='gray')
#            plt.show()
#            plt.imshow(AFimage, cmap='gray')
#            plt.show()
#            plt.imshow(Mop_image, cmap='gray')
#            plt.show()

        frame_flat = np.array(Mop_image).flatten()
        
#        frame_flat = np.array(AFimage).flatten()
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
        n=n+1
    trueArray = np.asarray(trueVector)
    predArray = np.asarray(predVector)
    precision, recall,_ ,_ = PRFmetrics(trueArray, predArray,average='binary')
    # below these are the precision list and recall list for all alpha cases
    precision_list.append(precision)
    recall_list.append(recall)
precision_Array = np.asarray(precision_list)
recall_Array = np.asarray(recall_list)
# we use the presicion and recall list to calculate the AUC for one P case
AUC_p = auc(recall_Array, precision_Array) 

precision_mop=precision_Array
recall_mop=recall_Array

AUC_mop = auc(recall_Array, precision_Array)



plt.plot(recall_Array,precision_Array,label='Video Stabilization BM | AUC =0,68 ')
plt.plot(recall_Array0,precision_Array0,label='With Moprhological filters | AUC=0,62')
plt.plot(recall_Array1,precision_Array1,label='Other Method | AUC=0,52')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Traffic')
plt.legend()
plt.savefig('traffic1.png')
plt.show()



