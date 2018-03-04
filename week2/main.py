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

from frameworkJFM import gaussian1D as g
from frameworkJFM import Original as o
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    print 'in00'+str(i)+'.jpg loaded'

#animacion = o('highway_im',data_dir,gt_dir)
#animacion.animacion(frame_list)
#print 'done!'
##Defining a class for highway dataset
highway = g('highway',data_dir,gt_dir,False)
#print highway.name
##Gaussian-based model--> compute
highway.get_1D(frame_list)


fig, axes = plt.subplots(nrows=2, ncols=1)
im = axes[0].imshow(highway.mean.astype(int), vmin=0, vmax=255)
axes[1].imshow(highway.std.astype(int), vmin=0, vmax=255)
cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(im, cax=cax, **kw)
for i in range(len(axes)):
    axes[i].axes.get_xaxis().set_visible(False)
    axes[i].axes.get_yaxis().set_visible(False)
axes[0].set_title('Mean values')
axes[1].set_title('Std values')
plt.savefig(highway.name+'_mean_stdplot.png', bbox_inches='tight', pad_inches = 0)
#print 'Reading to-motion-estimate files:...'
#for i in range(1050,1350):
#    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
#    print 'in00'+str(i)+'.jpg ...loaded!'
#    image = cv2.imread(im_dir,-1)
#    #Computing motion based in Gaussian1D
#    foreground = highway.get_motion(image,2)
#    #Create results folder before running code inside your working folder
#    #Saving results (binary) 1=foreground 0 = background
#    cv2.imwrite('results_highway/00'+str(i)+'.png',foreground)
#
##y,x= highway.plotF1vsth(frame_list,gt_list,np.arange(0,10,0.25))
#
#
#
##Fall dataset
#data_dir= '/media/jfm/Slave/Data/datasets/fall/input'
#gt_dir =   '/media/jfm/Slave/Data/datasets/fall/groundtruth'
##Instance deffinition  Fall dataset
#fall = g('fall_dataset',data_dir,gt_dir)
#print 'Defining '+fall.name+' instance'
#
#frame_list = []
#gt_list = []
#print 'Reading background modeling files:...'
#for i in range(1460,1560):
#    frame_list.append('in00'+str(i)+'.jpg')
#    gt_list.append('gt00'+str(i)+'.png')
#    print 'in00'+str(i)+'.jpg loaded'
#print 'done!'
#
##Gaussian-based model--> compute
#fall.get_1D(frame_list)
#print 'Reading to-motion-estimate files:...'
#for i in range(1460,1560):
#    im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
#    print 'in00'+str(i)+'.jpg ...loaded!'
#    image = cv2.imread(im_dir,-1)
#    #Computing motion based in Gaussian1D
#    foreground = fall.get_motion(image,2)
#    #Create results folder before running code inside your working folder
#    #Saving results (binary) 1=foreground 0 = background
#    cv2.imwrite('results_fall/for00'+str(i)+'.png',foreground)
#
##y_fall,x_fall= fall.plotF1vsth(frame_list,gt_list,np.arange(0,10,0.25))
#
##Traffic camera jitter dataset
#data_dir = '/media/jfm/Slave/Data/datasets/traffic/input'
#gt_dir =   '/media/jfm/Slave/Data/datasets/traffic/groundtruth'
##Instance deffinition  Fall dataset
#traffic = g('traffic_camjittering',data_dir,gt_dir)
#print 'Defining '+traffic.name+' instance'
#
#frame_list = []
#gt_list = []
#print 'Reading background modeling files:...'
#for i in range(950,1050):
#    if i >=1000:
#        frame_list.append('in00'+str(i)+'.jpg')
#        gt_list.append('gt00'+str(i)+'.png')
#        print 'in00'+str(i)+'.jpg loaded'
#    else:
#        frame_list.append('in000'+str(i)+'.jpg')
#        gt_list.append('gt000'+str(i)+'.png')
#        print 'in000'+str(i)+'.jpg loaded'       
#print 'done!'
#
##Gaussian-based model--> compute
#traffic.get_1D(frame_list)
#print 'Reading to-motion-estimate files:...'
#for i in range(950,1050):
#    if i >= 1000:
#        im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
#        print 'in00'+str(i)+'.jpg ...loaded!'
#    else:
#        im_dir = os.path.join(data_dir, 'in000'+str(i)+'.jpg')
#        print 'in000'+str(i)+'.jpg ...loaded!'
#    image = cv2.imread(im_dir,-1)
#    #Computing motion based in Gaussian1D
#    foreground = traffic.get_motion(image,2)
#    #Create results folder before running code inside your working folder
#    #Saving results (binary) 1=foreground 0 = background
#    cv2.imwrite('results_traffic/for00'+str(i)+'.png',foreground)

#y_traffic,x_traffic= traffic.plotF1vsth(frame_list,gt_list,np.arange(0,10,0.25))


#Plotting
#x=np.loadtxt('F1alpha_x.txt')
#y=np.loadtxt('F1alpha_y.txt')
#x_fall=np.loadtxt('x_fall.txt')
#y_fall=np.loadtxt('y_fall.txt')
#plt.plot(x,y,label='highway frames 1050-1350')
#plt.plot(x_fall,y_fall,label='fall frames 1460-1560')
#plt.plot(x_traffic,y_traffic,label='jitter frames 950-1050')
#plt.xlabel('alpha')
#plt.ylabel('F1 score')
#plt.title('F1 score vs alpha')
#plt.legend()
#plt.savefig('F1vsalpha.png')
print 'done! Press q to quit'
if cv2.waitKey(10) == ord('q'):
    print 'Bye'
