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

data_dir = '/media/jfm/Slave/Data/datasets/highway/input'
gt_dir   = '/media/jfm/Slave/Data/datasets/highway/groundtruth'

from setup import gaussian1D as g
import cv2
import os
print 'MSc in Computer Vision Barcelona'
print 'Universidad Politecnica de Catalunya'
print '......................................'
print '......................................'
#Creating a list of frame names to perform a background model
frame_list = []
gt_list = []
print 'Reading background modeling files:...'
for i in range(1050,1350):
    frame_list.append('in00'+str(i)+'.jpg')
    gt_list.append('gt00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg'
print 'done!'
#Defining a class for highway dataset
highway = g('highway_set',data_dir,gt_dir,False)
print highway.name
#Gaussian-based model--> compute
highway.get_1D(frame_list)
print 'Reading to-motion-estimate files:...'
for i in range(1050,1350):
    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
    print 'in00'+str(i)+'.jpg'
    image = cv2.imread(im_dir,-1)
    #Computing motion based in Gaussian1D
    foreground = highway.get_motion(image,2)
    #Create results folder before running code inside your working folder
    #Saving results (binary) 1=foreground 0 = background
    cv2.imwrite('results/res00'+str(i)+'.png',foreground)

precision, recall, f1_score = highway.evaluateSeveralFrames(frame_list,gt_list)
print 'done! Press q to quit'
if cv2.waitKey(10) == ord('q'):
    print 'Bye'
