import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from readFlowFile import read as read
import os
import matplotlib.pyplot as plt
import perspective
import math
from os import listdir
from os.path import isfile, join
option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.6,
    'gpu': 1.0
}

tfnet = TFNet(option)
capture = cv2.VideoCapture('input.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
class Car:
    def __init__(self,idn):
        self.id=idn
def IoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou
def Tracking(old_frame,new_frame,idn,framen,th=0.5,recursividad=5):
    #Updating dictionary with id for tracking. Old frame is supposed to be added
    desde = framen-recursividad-1
    if desde<0:
        desde = 0
    vector = range(desde,framen-1)
    vector.reverse()
    for i in range(len(new_frame)):
        new_frame[i]['id']=None
        new_frame[i]['color']=None
        new_frame[i]['tracked']=False
        new_frame[i]['speed']=[]
        

        for j in vector:
            IoU_list = []
            if new_frame[i]['tracked'] == False:
                for result in old_frame[j]:
                    boxA = [result['topleft']['x'], result['topleft']['y'],result['bottomright']['x'], result['bottomright']['y']]
                    boxB = [new_frame[i]['topleft']['x'], new_frame[i]['topleft']['y'],new_frame[i]['bottomright']['x'], new_frame[i]['bottomright']['y']]
                    IoU_list.append(IoU(boxA,boxB))
                if (not IoU_list) == False:
                    IoU_list = np.asarray(IoU_list)
                    ind = np.unravel_index(np.argmax(IoU_list, axis=None), IoU_list.shape)
                    ind=ind[0]
                    max_IoU = IoU_list[ind]
                    if max_IoU>th:
                        new_frame[i]['id']=old_frame[j][ind]['id']
                        new_frame[i]['color'] = old_frame[j][ind]['color']
                        new_frame[i]['tracked'] = True
                        new_frame[i]['speed'] = old_frame[j][ind]['speed']
                    elif j == vector[len(vector)-1]:
                        idn=idn+1
                        new_frame[i]['id']=idn
                        new_frame[i]['tracked'] = True
                        new_frame[i]['color'] = tuple(255 * np.random.rand(3))
                elif ((not IoU_list) == True) & (j == vector[len(vector)-1]):
                    idn=idn+1
                    new_frame[i]['id']=idn
                    new_frame[i]['tracked'] = True
                    new_frame[i]['color'] = tuple(255 * np.random.rand(3))
    return idn,new_frame
def Speed(old_frame,pts,frame,frame_anterior,rate,fps):
    frame = np.asarray(frame)
    mask = np.zeros(shape=frame.shape[0:2])
    for dic in old_frame:
        mask[dic['topleft']['y']:dic['bottomright']['y'],dic['topleft']['x']:dic['bottomright']['x']]=dic['id']
    mask_h, M = perspective.four_point_transform(mask, pts)
    frame_h, M = perspective.four_point_transform(frame, pts)
    frame_ah, M = perspective.four_point_transform(frame_anterior, pts)
    for i in range(len(old_frame)):
        binary = (mask_h==old_frame[i]['id'])
        cv2.imwrite('old.png',frame_ah)
        cv2.imwrite('new.png',frame_h)
        os.system('./deepflow2 old.png new.png sintel.flo')
        file = 'sintel.flo'
        flow = read(file)
        mod = np.linalg.norm(flow,axis=2)
        plt.imsave('opticalflow/'+str(framen-1)+'.png',mod)
        velocidad = float(np.mean(mod[binary==True])*fps/rate*3.6)
        if (velocidad > (np.mean(old_frame[i]['speed'])/2)) | (not old_frame[i]['speed']):
            old_frame[i]['speed'].append(velocidad)
        else:
            old_frame[i]['speed'].append(np.mean(old_frame[i]['speed']))
    return old_frame,M
def Centro(diccionario):
    tl = (diccionario['topleft']['x'], diccionario['topleft']['y'])
    br = (diccionario['bottomright']['x'], diccionario['bottomright']['y'])
    center  = ((br[0]-tl[0])/2,(br[1]-tl[1])/2)
    center  = (center[0]+tl[0],center[1]+tl[1])
    return center
def p(point):
    return (point[0],point[1],1)
def Colision(old_frame,pts,rate,M):
    centros = []
    road_dir1 = (pts[3]-pts[2])
    road_dir1 = road_dir1/np.linalg.norm(road_dir1)
    road_dir2 = (pts[3]-pts[2])
    road_dir2 = road_dir2/np.linalg.norm(road_dir2)
    for dic in old_frame:
        centros.append(Centro(dic))
    al = np.zeros(shape=(len(old_frame),len(old_frame)),dtype = bool)
    crit = np.zeros(shape=(len(old_frame),len(old_frame)))
    for i in range(len(old_frame)):
        for j in range(i-1):
            center_dir = (centros[i][0]-centros[j][0],centros[i][1]-centros[j][1])
            angle = abs(np.dot(center_dir,road_dir1)/np.linalg.norm(center_dir))
            angle2 = abs(np.dot(center_dir,road_dir2)/np.linalg.norm(center_dir))
            angle = math.acos(angle)*180/np.pi
            angle2 = math.acos(angle2)*180/np.pi
            if (angle<25) | (angle2 <25):
                #dist = np.linalg.norm(center_dir)
                p1 = np.matrix.dot(M,p(centros[i]))
                p1=p1/p1[2]
                p2 = np.matrix.dot(M,p(centros[j]))
                p2 = p2/p2[2]
                dist = np.linalg.norm(p1-p2)/rate
                v1 = old_frame[i]['speed']
                v2 = old_frame[j]['speed']
                v1=v1[0]
                v2=v2[0]
                d1 = v1*v1/16
                d2 = v2*v2/16
                r1 = 0.5*v1
                r2 = 0.5*v2
                print 'dist',str(dist)
                print 'D1',str(r1 + d1-dist-d2),'D2',str(r2+d2-dist-d1)
                criteria = min(r1 + d1-dist-d2,r2+d2-dist-d1) 
                al[i][j] = True
                crit[i][j] = criteria
    return crit, al
                    
                
            
colorbook = {
            'red':(0,0,255),
            'green':(0,255,0),
            'orange':(51,153,255)
            }
fps = capture.get(cv2.CAP_PROP_FPS)
idn = 0
framen=0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (320,240))
pts = np.array([(188,15), (1, 164), (290,239), (281, 15)])
rate = 1.78
mypath = '/home/jfm/Nets/darkflow/uab'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for frame in sorted(onlyfiles):
        framen=framen+1
        results = tfnet.return_predict(frame)
        
        #Case of 1st frame, initialise
        if idn == 0:
            old_frame = []
            cv2.imwrite('old.png',frame)
            for i in range(len(results)):
                idn = idn+1
                frame_anterior = frame
                results[i]['id'] = idn
                results[i]['color'] = tuple(255 * np.random.rand(3))
                results[i]['tracked'] = True
                results[i]['speed']=[]
            old_frame.append(results)
        else:
            cv2.imwrite('new.png',frame)
            idn,new_frame = Tracking(old_frame,results,idn,framen,th=0.5,recursividad=5)
            old_frame.append(new_frame)
            old_frame[framen-2],M=Speed(old_frame[framen-2],pts,frame,frame_anterior,rate,fps)
            criterio,alineacion = Colision(old_frame[framen-2],pts,rate,M)
            Cxy = []
            for i in range(len(old_frame[framen-2])):
                Cxy.append(Centro(old_frame[framen-2][i]))
            for i in range(len(old_frame[framen-2])):
                for j in range(i-1):
                    if alineacion[i][j] == True:
                        if criterio[i][j]>0:
                            frame_anterior = cv2.line(frame_anterior,Cxy[i],Cxy[j],colorbook['red'])
                        elif (criterio[i][j]<=0) & (criterio[i][j]>-10):
                            frame_anterior = cv2.line(frame_anterior,Cxy[i],Cxy[j],colorbook['red'])
                        else:
                            frame_anterior = cv2.line(frame_anterior,Cxy[i],Cxy[j],colorbook['green'])
            for result in old_frame[framen-2]:
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                frame_anterior = cv2.rectangle(frame_anterior, tl, br, result['color'], 2)
                frame_anterior = cv2.putText(frame_anterior, str(result['id'])+', '+str(int(np.mean(result['speed']))), tl, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)            
            cv2.imshow('frame', frame_anterior)
            out.write(frame_anterior)
            cv2.imwrite('results/frame'+str(framen)+'.png',frame_anterior)
            frame_anterior = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
out.release()
