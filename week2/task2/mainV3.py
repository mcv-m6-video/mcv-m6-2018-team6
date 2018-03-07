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
from matplotlib import cm
from frameworkJFMV2 import adaptative as g
from frameworkJFMV2 import Original as o
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
print 'MSc in Computer Vision Barcelona'
print 'Universidad Politecnica de Catalunya'
print '......................................'
print '......................................'


#data_dir = '/media/jfm/Slave/Data/datasets/highway/input'
#gt_dir   = '/media/jfm/Slave/Data/datasets/highway/groundtruth'
####Creating a list of frame names to perform a background model
#frames_train = []
#gt_train = []
#
#frames_test = []
#gt_test = []
#
#print 'Reading background modeling files:...'
#for i in range(1050,1201):
#    frames_train.append('in00'+str(i)+'.jpg')
#    gt_train.append('gt00'+str(i)+'.png')
#    print 'in00'+str(i)+'.jpg loaded'
#for i in range(1201,1351):
#    frames_test.append('in00'+str(i)+'.jpg')
#    gt_test.append('gt00'+str(i)+'.png')
#    print 'in00'+str(i)+'.jpg loaded'
#print 'Done!'
#
####Defining an instance for highway dataset
#
#highway = g('highway_gray',data_dir,gt_dir,'gray')
#print 'Instance ' +highway.name +' created'
#
###Compute background model (gaussian-based)
#highway.get_1D(frames_train)
###Plot Mean and Std
#highway.PlotMeanStd()
#plt.close()
#### Computing metric vs alpha
#highway.allvsalpha(frames_test,gt_test,np.arange(0,10.25,0.25))
#highway.saveAllvsalpha()
#print 'Reading to-motion-estimate files:...'
#results_list_dir = []
#for i in range(1201,1351):
#    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
#    print 'in00'+str(i)+'.jpg ...loaded!'
#    image = cv2.imread(im_dir,-1)
#    #Computing motion based in Gaussian1D
#    foreground = highway.get_motion(image,highway.x[np.argmax(highway.F1_vector)])
#    #Create results folder before running code inside your working folder
#    #Saving results (binary) 1=foreground 0 = background
#    cv2.imwrite('results_highway_gray/00'+str(i)+'.png',foreground)
#    results_list_dir.append('results_highway_gray/00'+str(i)+'.png')
#
#### Creating gif of RGB images + FP + FN
#highway.errorPainting(frames_test,gt_test,results_list_dir)
#
#### Creating a gif of motion estimation results
#animar = o('highway_gif_gray','results_highway_gray',gt_dir)
#res_list = []
#for i in range(1201,1351):
#    res_list.append('00'+str(i)+'.png')
#animar.animacion(res_list)
#
##### ===========================================================================
##### ===========================================================================
data_dir = '/media/jfm/Slave/Data/datasets/fall/input'
gt_dir   = '/media/jfm/Slave/Data/datasets/fall/groundtruth'
###Creating a list of frame names to perform a background model
frames_train = []
gt_train = []

frames_test = []
gt_test = []

print 'Reading background modeling files:...'
for i in range(1460,1511):
    frames_train.append('in00'+str(i)+'.jpg')
    gt_train.append('gt00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg loaded'
for i in range(1511,1561):
    frames_test.append('in00'+str(i)+'.jpg')
    gt_test.append('gt00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg loaded'
print 'Done!'

###Defining an instance for fall dataset

fall = g('fall_ad',data_dir,gt_dir,'gray')
print 'Instance ' +fall.name +' created'

##Compute background model (gaussian-based)
fall.get_1D(frames_train)
##Plot Mean and Std
fall.PlotMeanStd()
plt.close()
### Computing metric vs alpha
fall.allvsalpha(frames_test,gt_test,np.arange(0,6,1),np.arange(0,1,0.2))
fall.saveAllvsalpha()
print 'Reading to-motion-estimate files:...'
results_list_dir = []
for i in range(1511,1561):
    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
    print 'in00'+str(i)+'.jpg ...loaded!'
    image = cv2.imread(im_dir,-1)
    #Computing motion based in Gaussian1D
    indice,dontcare = np.unravel_index(np.argmax(fall.F1_vector, axis=None), fall.F1_vector.shape)
    foreground = fall.get_motion(image,[fall.alpha[indice],fall.beta[dontcare]])
    #Create results folder before running code inside your working folder
    #Saving results (binary) 1=foreground 0 = background
    cv2.imwrite('results_fall/00'+str(i)+'.png',foreground)
    results_list_dir.append('results_fall/00'+str(i)+'.png')

### Creating gif of RGB images + FP + FN
fall.errorPainting(frames_test,gt_test,results_list_dir)

### Creating a gif of motion estimation results
animar = o('fall_gif_gray','results_fall_gray',gt_dir)
res_list = []
for i in range(1511,1561):
    res_list.append('00'+str(i)+'.png')
animar.animacion(res_list)

alpha, beta = np.meshgrid(fall.alpha, fall.beta)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(alpha, beta, fall.F1_vector, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#### ===========================================================================
#### ===========================================================================
#data_dir = '/media/jfm/Slave/Data/datasets/traffic/input'
#gt_dir   = '/media/jfm/Slave/Data/datasets/traffic/groundtruth'
####Creating a list of frame names to perform a background model
#frames_train = []
#gt_train = []
#
#frames_test = []
#gt_test = []
#
#for i in range(950,1001):
#    if i >=1000:
#        frames_train.append('in00'+str(i)+'.jpg')
#        gt_train.append('gt00'+str(i)+'.png')
#        print 'in00'+str(i)+'.jpg loaded'
#    else:
#        frames_train.append('in000'+str(i)+'.jpg')
#        gt_train.append('gt000'+str(i)+'.png')
#        print 'in000'+str(i)+'.jpg loaded'       
#print 'done!'
#print 'Reading background modeling files:...'
#for i in range(1001,1051):
#    frames_test.append('in00'+str(i)+'.jpg')
#    gt_test.append('gt00'+str(i)+'.png')
#    print 'in00'+str(i)+'.jpg loaded'
#print 'Done!'
#
####Defining an instance for fall dataset
#
#traffic = g('traffic_gray',data_dir,gt_dir,'gray')
#print 'Instance ' +traffic.name +' created'
#
###Compute background model (gaussian-based)
#traffic.get_1D(frames_train)
###Plot Mean and Std
#traffic.PlotMeanStd()
#plt.close()
#### Computing metric vs alpha
#traffic.allvsalpha(frames_test,gt_test,np.arange(0,10.25,0.25))
#traffic.saveAllvsalpha()
#print 'Reading to-motion-estimate files:...'
#results_list_dir = []
#for i in range(1001,1051):
#    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
#    print 'in00'+str(i)+'.jpg ...loaded!'
#    image = cv2.imread(im_dir,-1)
#    #Computing motion based in Gaussian1D
#    foreground = traffic.get_motion(image,traffic.x[np.argmax(traffic.F1_vector)])
#    #Create results folder before running code inside your working folder
#    #Saving results (binary) 1=foreground 0 = background
#    cv2.imwrite('results_traffic_gray/00'+str(i)+'.png',foreground)
#    results_list_dir.append('results_traffic_gray/00'+str(i)+'.png')
#
#### Creating gif of RGB images + FP + FN
#traffic.errorPainting(frames_test,gt_test,results_list_dir)
#
#### Creating a gif of motion estimation results
#animar = o('traffic_gif_gray','results_traffic_gray',gt_dir)
#res_list = []
#for i in range(1001,1051):
#    res_list.append('00'+str(i)+'.png')
#animar.animacion(res_list)


#### ===========================================================================
#### ===========================================================================
#### ===========================================================================
#### ===========================================================================

plt.plot(highway.x,highway.F1_vector,label='highway frames 1201-1350')
plt.plot(fall.x,fall.F1_vector,label='fall frames 1511-1560')
plt.plot(traffic.x,traffic.F1_vector,label='c jittering frames 1001-1051')
plt.xlabel('alpha')
plt.ylabel('F1 score')
plt.title('F1 score vs alpha')
plt.legend()
plt.savefig('F1vsalpha_gray_highway.png')
plt.close()

plt.plot(highway.recall_vector,highway.precision_vector,label='highway frames 1201-1350')
plt.plot(fall.recall_vector,fall.precision_vector,label='fall frames 1511-1560')
plt.plot(traffic.recall_vector,traffic.precision_vector,label='c jittering frames 1001-1051')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision - Recall curve')
plt.legend()
plt.savefig('PR_gray_highway.png')
