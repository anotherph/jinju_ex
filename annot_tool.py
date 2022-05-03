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

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:   # visible point 
        cv2.circle(data['image'],(x,y),3,(0,0,255),-1)
        cv2.imshow('image',data['image'])
        
        if len(data['points'])<4:
            data['points'].append([x,y,1])
    elif event == cv2.EVENT_RBUTTONDOWN: # invisible point
        data['points'].append([0,0,0])
    else:
        pass

def get_points(image):
    data ={}
    data['image']=image.copy()
    data['points']=[]
    
    cv2.imshow('image',image)
    cv2.setMouseCallback("image", mouse_handler,data)
    cv2.waitKey()
    
    points=np.array(data['points'],dtype=float)
    
    return points
 
if __name__ == "__main__":

    root_annos='/home/jekim/workspace/jinju_ex/data_original/annos_stand'
    root_image='/home/jekim/workspace/jinju_ex/data_original/image_stand'
    desti_annos='/home/jekim/workspace/jinju_ex/data_stand/annos'
    desti_image='/home/jekim/workspace/jinju_ex/data_stand/image'
    
    # for file_name_temp in os.listdir(root_annos):
        # file_name=file_name_temp.split('.')[0][:6]
    file_name='008860'
    name_annos = os.path.join(root_annos,file_name+'_keypoints.json')
    name_image = os.path.join(root_image,file_name+'.jpg')
    
    with open(name_annos) as f: annos = json.load(f)
    annos_edit=annos['people'][0]
    
    # change the the order of keypoints to being same as that of cocodataset
    keypoints_openpose=np.array(annos['people'][0]['pose_keypoints_2d']).reshape(-1,3)
    keypoints_COCO=keypoints_openpose[[0,14,15,16,17,2,5,3,6,4,7,8,11,8,12,10,13],:]
    
    # add the keypoints of knife
    annos_edit['pose_keypoints_2d']=keypoints_COCO.reshape(-1).tolist()
    annos_edit['knife']=list() # left front-end/left back-end /right front-end/right back-end
    
    image=cv2.imread(name_image)    
    points_src=get_points(image)
    annos_edit['knife']=points_src.reshape(-1).tolist()
    
    cv2.destroyAllWindows()
        
    with open(os.path.join(desti_annos,file_name+'_keypoints.json'), "w") as json_file:
        json.dump(annos_edit, json_file)
