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

from frameworkJFMV2 import gaussian1D as g
from frameworkJFMV2 import Original as o
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
print 'MSc in Computer Vision Barcelona'
print 'Universidad Politecnica de Catalunya'
print '......................................'
print '......................................'



##Images and Grountruth directory
data_dir = '/media/jfm/Slave/Data/datasets/traffic/input'
gt_dir   = '/media/jfm/Slave/Data/datasets/traffic/groundtruth'

###Defining an instance for traffic dataset

traffic = g('traffic_hsv',data_dir,gt_dir,'HSV')
##Load predefined images for training and testing
traffic.Read('traffic')

###Compute Background-foreground model
traffic.get_1D(traffic.frames_train)

### Plot Mean and Std (in case of non-adaptative model)
#traffic.PlotMeanStd()
#plt.close()

### Computing metric vs alpha from alpha =0 to alpha= 15 step 0.25
#traffic.allvsalpha(traffic.frames_test,traffic.gt_test,[2],np.arange(0,10.1,0.1))
#### Save metrics in text files
#traffic.saveAllvsalpha()
#traffic.LoadAllvsalpha()

results_list_dir = []
for i in range(1001,1051):
    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
    print 'in00'+str(i)+'.jpg ...loaded!'
    image = cv2.imread(im_dir,-1)
    #Computing motion based in Gaussian1D using alpha related to highest F1
    #foreground,shadow = traffic.get_motion(image,traffic.x[np.argmax(traffic.F1_vector)])
    foreground,shadow = traffic.get_motion(image,2,)
    #Create results folder before running code inside your working folder
    #Saving results (binary) 1=foreground 0 = background
    cv2.imwrite('results_traffic_gray/00'+str(i)+'.png',foreground)
    results_list_dir.append('results_traffic_gray/00'+str(i)+'.png')

### Creating gif of RGB images + FP + FN
traffic.errorPainting(traffic.frames_test,traffic.gt_test,results_list_dir)

### Creating a gif of motion estimation results
animar = o('traffic_gif_gray','results_traffic_gray',gt_dir)
res_list = []
for i in range(1001,1051):
    res_list.append('00'+str(i)+'.png')
animar.animacion(res_list)


