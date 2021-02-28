# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:32:04 2020

@author: Conor
"""
import cv2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import transform
import imageio
datagen = ImageDataGenerator(rescale=1./255)
def cleanupimage(np_image):
   np_image = np.array(np_image).astype('float32')/255
   np_image = np.expand_dims(np_image, axis=0)
   return np_image
model = keras.models.load_model(r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\video100itcleandir.h5')

GREEN = [0,255,0]
RED = [0,0,255]
BLUE = [255,0,0]
colors = [GREEN,RED,BLUE]
cap = cv2.VideoCapture(r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\Video 1099.wmv')
fps=15
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
mo1 = cv2.VideoWriter(r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\mo1.mp4', fourcc, fps,(208,360))
mo2 = cv2.VideoWriter(r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\mo2.mp4', fourcc, fps,(208,360))
mo3  =cv2.VideoWriter(r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\mo3.mp4', fourcc, fps,(208,360))
vid_writers = [mo1,mo2,mo3]                                          
output=[]
z_out_out = []
i=0
FPS =  cap.get(cv2.CAP_PROP_FPS)
writer= imageio.get_writer(r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\mo1.gif', mode='I') 
while(cap.isOpened()):
    i+=1
    ret,frame = cap.read()
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)
    #if i == 30:
    #    break
    #if cv2.waitKey(1) & 0xFF ==ord('q'):
    #    break
    m1 = frame[:,0:208]
    m2 = frame[:,208:416]
    m3 = frame[:,416:624]
    frames = [m1,m2,m3]
    tout=[]
    z_out=[]
    for fnn,fr in enumerate(frames):
        out = model.predict(cleanupimage(fr))
        tout.append(out)
        #z = int(np.round(out))
        z = int(np.argmax(out))
        z_out.append(z)
        new_frame = fr
        cir = cv2.circle(new_frame,(10,10),10,colors[z],-1)
        vid_writers[fnn].write(cir)
        if fnn==0 and i>150 and i<300:
            writer.append_data(fr)
        if fnn==0 and i==300:
            writer.close()
    output.append(tout)
    z_out_out.append(z)
    if  i/FPS>(10*60):
        break
    if i%300 ==0:
        
        print(i/FPS)
mo1.release()
mo2.release()
mo3.release()