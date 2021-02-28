# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:53:16 2020

@author: Conor
"""

import os,shutil
import random

folders = ['top','bottom','none']

original = r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\all_images'
output = r'C:\Users\Conor\Desktop\TRNDeepnetWorkingdir\AL_trn files\run_dir'
output_train = os.path.join(output,'train')
output_validation = os.path.join(output,'validation')
os.mkdir(output_train)
os.mkdir(output_validation)

for folder in folders:
    base = os.path.join(original,folder)
    target_validation = os.path.join(output_validation,folder)
    target_train = os.path.join(output_train,folder)
    os.mkdir(target_validation)
    os.mkdir(target_train)
    files = os.listdir(base)
    random.shuffle(files)
    for filen in files[0:1300]:
        src = os.path.join(base,filen)
        dst = os.path.join(target_train,filen)
        shutil.copyfile(src,dst)
    for filen in files[1301:2000]:
        src = os.path.join(base,filen)
        dst = os.path.join(target_validation,filen)
        shutil.copyfile(src,dst)
    
                       