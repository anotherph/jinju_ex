#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:44:43 2022

@author: jekim

seperate the whole json-file into each piece of json-file 

"""

import os
import sys 
import cv2 as cv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage import io, transform
from skimage.color import rgb2gray
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import json

json_path = "/media/jekim/Samsung_T5/COCO2017/annotations/person_keypoints_val2017.json"
root_image = "/media/jekim/Samsung_T5/COCO2017/val2017/"
root_annos=json_path 
save_dir = "/home/jekim/Documents/valid/"

with open(root_annos) as f: annos = json.load(f)
a=0

for idx in range(len(annos['annotations'])):
    # print(idx)

    file_name=annos['annotations'][idx]['image_id']
    landmarks=np.array(annos['annotations'][idx]['keypoints'])
    bbox = annos['annotations'][idx]['bbox']
    
    annos_edit={} # declare new dict 
    annos_edit['id']='{:012d}'.format(file_name)
    annos_edit['keypoints']=annos['annotations'][idx]['keypoints']
    annos_edit['bbox']=annos['annotations'][idx]['bbox']
    
    save_json=os.path.join(save_dir,str(a)+'.json')
    
    name_img=os.path.join(root_image,'{:012d}'.format(file_name)+'.jpg')
    image = io.imread(name_img)
    
    if sum(annos_edit['keypoints'])==0: # no keypoint
        pass
    elif len(image.shape)!=3: # in case of grey image
        pass
    else:
        with open(save_json, "w") as json_file:
            json.dump(annos_edit, json_file)
        a+=1
    
    # if os.path.isfile(save_json):
    #     print("error", a)
    #     a+=1
    # else:
    #     with open(save_json, "w") as json_file:
    #         json.dump(annos_edit, json_file)
 
# json_dir='/home/jekim/workspace/Deepfashion2_Training/Deepfashion2_Training/dataset2_op_nooclu/validation/annos/013951.json'
# with open(json_dir) as f: annos = json.load(f)