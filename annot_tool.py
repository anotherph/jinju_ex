#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:43:29 2022

@author: jekim

data annotation tool 
1. read json-file
2. open the image 
3. click the point 
4. write json-file including new points

"""
import os
import sys 
import cv2 
import json
import numpy as np
import matplotlib.pyplot as plt

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables

    count=0
    if event == cv2.EVENT_LBUTTONDOWN:
        if count==0:
            refPt = [(x, y)]
            count+=1
        else:
            refPt.append((x, y))
            count+=1
            
    return refPt
 
if __name__ == "__main__":

    root_annos='/home/jekim/workspace/jinju_ex/data_original/annos_left'
    root_image='/home/jekim/workspace/jinju_ex/data_original/image_left'
    desti_annos=''
    
    # for file_name_temp in os.listdir(root_annos):
    # file_name=file_name_temp.split('.')[0][:6]
    file_name='002700'
    name_annos = os.path.join(root_annos,file_name+'_keypoints.json')
    name_image = os.path.join(root_image,file_name+'.jpg')
    
    with open(name_annos) as f: annos = json.load(f)
    annos_edit=annos['people'][0]
    annos_edit['knife']=list() # left front-end/left back-end /right front-end/right back-end
    
    image=cv2.imread(name_image)
    clone=image
    
    # global refPt

    while True:
        cv2.imshow("image",image)
        cv2.setMouseCallback("image", click_and_crop)
        key=cv2.waitKey(1) & 0xFF
        
        if key == ord("r"):
            image = clone
        elif key == ord("c"):
            break

    print(refPt)
        
        # with open("student_file.json", "w") as json_file:
        # json.dump(student_data, json_file)

