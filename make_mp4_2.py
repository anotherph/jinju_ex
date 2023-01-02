#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 19:37:11 2022

@author: jekim
"""

# original, smpl (smplx) 결과 합친것 

import re
import os
import numpy as np

# os.system(f'ffmpeg -r 30 -i jekim7/output/smpl.*.png'
#               f' -crf 30 jekim7/output/smpl.mp4')

# pathIn1='/home/jekim/workspace/jinju_ex/data/0720_SGU/original_video/zed_camera'      
pathIn1='/home/jekim/workspace/jinju_ex/data/0720_SGU/original_video/zed_camera/geommu3/result_mask'       
 
pathIn_1 = os.path.join(pathIn1,'result')
pathIn_2 = os.path.join(pathIn1,'geommu4/result/')
pathOut =os.path.join(pathIn1,'result/video.mp4')

paths1 = [os.path.join(pathIn_1 , i ) for i in os.listdir(pathIn_1) if re.search(".png$", i )]
paths2 = [os.path.join(pathIn_2 , i ) for i in os.listdir(pathIn_1) if re.search(".png$", i )]

## 정렬 작업
paths1 = list(np.sort(paths1))
paths2 = list(np.sort(paths2))

#len('ims/2/a/2a.2710.png')
#pathIn= './jekim7/output/smpl/'

fps = 20
import cv2
frame_array = []
for idx in range(len(paths1)) : 

    img1 = cv2.imread(paths1[idx])
    img2 = cv2.imread(paths2[idx])

    # img_t= cv2.vconcat([img2,img3]) #original images
    # img= cv2. hconcat([img1,img2])
    img= cv2.hconcat([img1,img2])
    
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()