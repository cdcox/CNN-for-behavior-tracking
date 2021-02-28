# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:57:58 2020

@author: Conor
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

label_decode=['none','top','bottom','errors','trash']
cap = cv2.VideoCapture(r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\Video 1099.wmv')
directory = r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files'
out_dir = r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\all_images'
for label in label_decode:
    os.mkdir(os.path.join(out_dir,label))
file_list = os.listdir(directory)
#note TRNs must be ordered by putting nubmers in front or somehting
trns = [x for x in file_list if '.trn' in x]
prepped_trns = []
for trn in trns:
    with open(os.path.join(directory,trn),"r") as f:
        z = f.readlines()
    z = [x.rstrip("\n") for x in z]
    z = z[8:-9]
    out_array = np.zeros([int(len(z)/3),3])
    for znn, line in enumerate(z[::3]):
        if 'LKeyDown' in line:
            out_array[znn,0]=1
        if 'RKeyDown' in line:
            out_array[znn,0] = 2
        start = line.split(':')[1]
        line2 = z[znn*3+1]
        stop = line2.split(':')[1]
        out_array[znn,1] = float(start)*2
        out_array[znn,2] = float(stop)*2
    prepped_trns.append(out_array)
FPS =  cap.get(cv2.CAP_PROP_FPS)
i = 0
m1_start = 2
m2_start = 3
m3_start = 15
starts = [m1_start,m2_start,m3_start]
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
    for fnn,fr in enumerate(frames):
        rel_time = i/FPS-starts[fnn]
        temp_arr = prepped_trns[fnn]
        star_check = (temp_arr[:,1])<rel_time
        stop_check = (temp_arr[:,2])>rel_time
        check_array = star_check*stop_check 
        buffstar_check = (temp_arr[:,1]-7/FPS)<rel_time
        buffstop_check = (temp_arr[:,2]+7/FPS)>rel_time
        buff_check_array = buffstar_check*buffstop_check
        if np.sum(check_array)==1:
            row = np.where(check_array)[0][0]
            label = temp_arr[row,0]
        elif np.sum(buff_check_array)>=1:
            label = 4
        elif np.sum(check_array)>1:
            print('somethin is wrong '+str(i))
            label = 3
        else:
            label = 0
        label = int(label)
        cv2.imwrite(os.path.join(out_dir,label_decode[label],label_decode[label]+str(fnn)+'_'+str(i)+'.png'),fr)
    if i%100 ==0 :
        print(i/FPS)
    if  i/FPS>(10*60):
        break
        
        
    
    
cap.release()
cv2.destroyAllWindows()

