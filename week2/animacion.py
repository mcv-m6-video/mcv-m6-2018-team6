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
import matplotlib.image as mgimg
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
data_dir='/home/jfm/Video Analysis/GitHub/mcv-m6-2018-team6/week2/results_highway'
fig = plt.figure()
plt.axis('off')

ims = []
for i in range(1050,1350):
    im_dir = os.path.join(data_dir, 'for00'+str(i)+'.png')
    print 'in00'+str(i)+'.jpg ...loaded!'
    im = mgimg.imread(im_dir)
    imgplot = plt.imshow(im)
    ims.append([imgplot])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# ani.save('dynamic_images.mp4')

plt.show()
ani.save('highway.gif')