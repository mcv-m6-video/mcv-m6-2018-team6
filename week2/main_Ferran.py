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

data_dir = '/home/ferran/Desktop/M6_project/datasets/highway/input'
gt_dir   = '/home/ferran/Desktop/M6_project/datasets/highway/groundtruth'

from frameworkJFM import gaussian1D as g
from frameworkJFM import Original as o
#from frameworkJFM import evaluateSeveralFrames as evaluateSeveralFrames
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

print 'MSc in Computer Vision Barcelona'
print 'Universidad Politecnica de Catalunya'
print '......................................'
print '......................................'
###Creating a list of frame names to perform a background model
def main():
    
    data_dir = '/home/ferran/Desktop/M6_project/datasets/highway/input'
    gt_dir   = '/home/ferran/Desktop/M6_project/datasets/highway/groundtruth'
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
        


    #animacion = o('highway_im',data_dir,gt_dir)
    #animacion.animacion(frame_list)
    highway = g('highway',data_dir,gt_dir,False)    
    highway.get_1D(frames_train)
    search_grid(frames_train, frames_test, gt_test,data_dir,gt_dir)    #print 'done!'
    ##Defining a class for highway dataset
    
#    print highway.name
    #Gaussian-based model--> compute

    #highway.get_motion(frames_train,1.8)

#highway.PlotMeanStd()
#print 'Reading to-motion-estimate files:...'
#range_th=10
#step_th=5
#range_rho=1
#step_rho=0.5
#results_list_dir = []
#F1_list=np.zeros(shape=(int(range_th/step_th),int(range_rho/step_rho)))
#count_th=0
#count_rho=0
#first_one=0
#for th_aux in range(0, range_th*10, int(step_th*10)):
#    count_rho=0
#    for rho_aux in range(0, range_rho*100, int(step_rho*100)):
#        th=th_aux/10
#        rho=rho_aux/100
#        first_one=first_one+1
#        precision, recall, F1 = highway.evaluateSeveralFrames_adaptative(frames_test,gt_test,th,rho,first_one)
#        F1_list[count_th,count_rho]=F1
#        count_rho=count_rho+1
#    count_th=count_th+1
#
#row,column = np.unravel_index(F1_list.argmax(), F1_list.shape)
#Best_F1=F1_list[row,column]        

def search_grid(frames_train, frames_test, gt_test,data_dir,gt_dir):
    alpha_vector = np.arange(1, 10, 2)
    rho_vector = np.arange(0, 0.5, 0.25)
    F1_matrix = np.zeros([len(alpha_vector),len(rho_vector)])
    for i in range(len(alpha_vector)):
        alpha = alpha_vector[i]
        for j in range(len(rho_vector)):
            rho = rho_vector[j]
            highway = g('highway',data_dir,gt_dir,False,alpha=alpha, rho=rho)
            image=highway.get_1D(frames_train)
            for i in range(1200,1351):
                im_dir = os.path.join(data_dir, 'in00'+str(i)+'.jpg')
                print 'in00'+str(i)+'.jpg ...loaded!'
                image = cv2.imread(im_dir,-1)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#                im = np.asarray(im)
                prediction = highway.get_motion_adaptative(frames_test,image)
            f1_score = g.evaluation_metrics(gt_test, prediction)
            F1_matrix[i,j] = f1_score
    axis = ["Ro","Alpha", "F1-score"]

    X, Y = np.meshgrid(rho_vector,alpha_vector)
    Z = F1_matrix

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    axis = ["Ro","Alpha", "F1-score"]
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])

#        plt.savefig('grid_search.png',dpi=300)
    plt.show()
    return F1_matrix,prediction


#highway.errorPainting(frame_list,gt_list,results_list_dir)
#animacion = o('highway_gif','results_highway',gt_dir)
res_list = []
for i in range(1050,1350):
    res_list.append('00'+str(i)+'.png')
#animacion.animacion(res_list)
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

##Gaussian-based model--> compute
#fall.get_1D(frame_list)
###PLOTTING MEAN AND STD AS IMAGES AND STORING EM
#fall.PlotMeanStd()
#plt.close()
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
if __name__ == "__main__":
    main()