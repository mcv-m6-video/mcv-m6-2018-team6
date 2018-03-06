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

##MASTER IN COMPUTER VISION
##Computer Vision Center Barcelona

from frameworkJFM import gaussian1D as g
from frameworkJFM import Original as o
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
print 'MSc in Computer Vision Barcelona'
print 'Universidad Politecnica de Catalunya'
print '......................................'
print '......................................'


data_dir = '/media/jfm/Slave/Data/datasets/highway/input'
gt_dir   = '/media/jfm/Slave/Data/datasets/highway/groundtruth'
###Creating a list of frame names to perform a background model
frames_train = []
gt_train = []

frames_test = []
gt_test = []

print 'Reading background modeling files:...'
for i in range(1050,1201):
    frames_train.append('in00'+str(i)+'.jpg')
    gt_train.append('gt00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg loaded'
for i in range(1201,1351):
    frames_test.append('in00'+str(i)+'.jpg')
    gt_test.append('gt00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg loaded'
print 'Done!'

###Defining an instance for highway dataset

highway = g('highway',data_dir,gt_dir,False)
print 'Instance' +highway.name +'created'

##Compute background model (gaussian-based)
highway.get_1D(frames_train)
##Plot Mean and Std
highway.PlotMeanStd()
plt.close()
### Computing metric vs alpha
highway.allvsalpha(frames_test,gt_test,np.arange(0,10,0.25))
highway.saveAllvsalpha()
print 'Reading to-motion-estimate files:...'
results_list_dir = []
for i in range(1201,1351):
    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
    print 'in00'+str(i)+'.jpg ...loaded!'
    image = cv2.imread(im_dir,-1)
    #Computing motion based in Gaussian1D
    foreground = highway.get_motion(image,highway.x[np.argmax(highway.F1_vector)])
    #Create results folder before running code inside your working folder
    #Saving results (binary) 1=foreground 0 = background
    cv2.imwrite('results_highway/00'+str(i)+'.png',foreground)
    results_list_dir.append('results_highway/00'+str(i)+'.png')

### Creating gif of RGB images + FP + FN
highway.errorPainting(frames_train,gt_train,results_list_dir)

### Creating a gif of motion estimation results
animacion = o('highway_gif','results_highway',gt_dir)
res_list = []
for i in range(1201,1351):
    res_list.append('00'+str(i)+'.png')
animacion.animacion(res_list)





### ===========================================================================




