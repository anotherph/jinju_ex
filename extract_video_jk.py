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

videoname='/home/jekim/workspace/jinju_ex/data/jinju_video.mp4'
outpath='/home/jekim/workspace/jinju_ex/data/image_right'

os.makedirs(outpath, exist_ok=True)
video = cv2.VideoCapture(videoname)
totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
start=2700 #start frame
end=9480 #end frame

for cnt in tqdm(range(totalFrames), desc='{:10s}'.format(os.path.basename(videoname))):
# for cnt in tqdm(range(start,end),desc='{:10s}'.format(os.path.basename(videoname))):
    ret, frame = video.read()
    if cnt < start:continue
    if cnt >= end:break
    if not ret:continue
    
    # cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt)), frame[100:550,250:550]) # for the left shot
    cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt)), frame[80:620,650:1280]) # for the right shot 

video.release()