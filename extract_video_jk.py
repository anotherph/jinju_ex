#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:05:57 2022

@author: jekim

extract images from video
"""

import os, sys
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

videoname='/home/jekim/workspace/jinju_ex/data_original/jinju_video.mp4'
outpath='/home/jekim/workspace/jinju_ex/data_original/image_stand'

os.makedirs(outpath, exist_ok=True)
video = cv2.VideoCapture(videoname)
totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
start=8488 #start frame
end=9480 #end frame
inv=10 # interval, save the image every 10 frame
for cnt in tqdm(range(totalFrames), desc='{:10s}'.format(os.path.basename(videoname))):
# for cnt in range(start,end,inv):
    ret, frame = video.read()
    if cnt < start:continue
    if cnt%inv !=0 : continue
    if cnt >= end:break
    if not ret:continue
    
    temp_frame=frame[100:550,250:550] # for the left shot
    # temp_frame=frame[80:620,650:1280] # for the right shot 
    factor=2
    img = cv2.resize(temp_frame, dsize=(temp_frame.shape[1]*factor, temp_frame.shape[0]*factor), interpolation=cv2.INTER_AREA)
    cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt)), img) 

video.release()

# openpose 결과
#./build/examples/openpose/openpose.bin --image_dir /home/jekim/workspace/jinju_ex/data_original/image_stand --write_json /home/jekim/workspace/jinju_ex/data_original/annos_stand
#./build/examples/openpose/openpose.bin --model_pose coco --image_dir /home/jekim/workspace/jinju_ex/data_original/image_stand --write_json /home/jekim/workspace/jinju_ex/data_original/annos_stand