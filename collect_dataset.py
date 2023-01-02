#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:13:38 2022

collecting the dataset

@author: jekim
"""

import os
import sys 
import cv2 
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil

path1_image='/home/jekim/workspace/jinju_ex/data/0720_SGU/original_video/iPhone/IMG_0138/images'
path1_annos='/home/jekim/workspace/jinju_ex/data/0720_SGU/original_video/iPhone/IMG_0138/annos'

dest_image='/home/jekim/workspace/jinju_ex/data/0720_SGU/dataset/images'
dest_annos='/home/jekim/workspace/jinju_ex/data/0720_SGU/dataset/annos'

# list_annos=os.listdir(path1_annos)
list_check=os.listdir(dest_annos)
list_check.sort()
list_=os.listdir(path1_annos)
list_.sort()

for ind in list_:
    temp_ind=int(ind[:5])+int(list_check[-1][:5])+10
    file_name='{0:05d}'.format(temp_ind)
    # file_name=ind[:5]
    
    file_name_ori=ind[:5]
        
    annos_ori=os.path.join(path1_annos,file_name_ori+'.json')
    image_ori=os.path.join(path1_image,file_name_ori+'.jpg')
    annos=os.path.join(dest_annos,file_name+'.json')
    image = os.path.join(dest_image,file_name+'.jpg') 
    
    shutil.copy(annos_ori, annos)
    shutil.copy(image_ori, image)

        

