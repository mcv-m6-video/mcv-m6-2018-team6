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

#Creating a list of frame name
frame_list = []
for i in range(1050,1350):
    frame_list.append('in00'+str(i)+'.jpg')

highway = g(False)
highway.get_1D(frame_list,data_dir)

for i in range(1050,1350):
    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
    image = cv2.imread(im_dir,-1)
    print 'i: ' +str(i)
    foreground = highway.get_motion(image,2)
    cv2.imwrite('results/res00'+str(i)+'.png',foreground)