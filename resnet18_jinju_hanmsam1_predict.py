#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

resnet18 to detect the keypoints of hansam

prediction (image only)
for code of resnet18_jinju_hansma1.py

load each pretrained network to detect left and right points
and predict the left/right respectively.
finally, print the left/right points on the original image together.  

@author: jekim
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
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import json
from pycocotools.coco import COCO 
import pylab

# %matplotlib inline

class HandCrop(object):
    " crop the area of the hand in image"
    
    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        # image = sample[:,:,:3]
    
        '''bbox = [x1,y1,x2,y2]'''        
        
        image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]        
        # landmarks = landmarks - [bbox[0], bbox[1]]
        
        # plt.scatter(landmarks[:,0], landmarks[:,1], c = 'r', s = 5)
        # plt.imshow(image)
        
        return {'image': image}

class Rescale_padding(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        
        '''rescale'''

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h < w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]
        
        '''padding'''
        
        h, w = img.shape[:2]
        # max_wh = np.max([w, h]) #padding
        max_wh = self.output_size+0 #padding to extended box
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5 # left
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5 # top
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5 # right
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5 # bottom
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        
        img_padded = np.ones((max_wh,max_wh,3))*(-1)
        img_padded[int(b_pad):int(b_pad)+h,int(l_pad):int(l_pad)+w,:]=img
        
        # landmarks = landmarks + [int(l_pad),int(b_pad)]
                

        return {'image': img_padded}

class Normalize(object):

    def __call__(self, sample):
        image = sample['image']
        
        '''normalize the image using mean & var'''
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5], [0.5])
            ])
        
        img_normalized = transform_norm(image)
        img=np.array(img_normalized)
        
        # landmarks = landmarks / [img.transpose(1,2,0).shape[1], img.transpose(1,2,0).shape[0]]
        
        # image = TF.normalize(image, [0.5], [0.5])
        
        # return {'image': img.transpose(1,2,0), 'landmarks': landmarks}
        
        return {'image': img_normalized}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        landmarks = landmarks / [image.shape[1], image.shape[0]]
        image = image.transpose((2, 0, 1))
        
        # plt.scatter(landmarks[:,0]*image.shape[0], landmarks[:,1]*image.shape[1], c = 'c', s = 5)
        # plt.imshow(image)
        # plt.show()
        
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
    
class Network(nn.Module):
    def __init__(self,num_classes=(6)*2):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x
    
    def cal_loss(self, predictions, landmarks_val, landmarks_vis):
        batch_size = predictions.shape[0]

        mask = landmarks_vis.reshape(batch_size*num_class*2,-1)
        predic = predictions.reshape(batch_size*num_class*2,-1)
        gt = landmarks_val.reshape(batch_size*num_class*2,-1)
                 
        loss = torch.pow(mask  * (predic - gt), 2).mean()

        return loss        
    
def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))   
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        
    sys.stdout.flush()
    
def landmark_Recursion(bbox, landmarks):
    # recursion of landmarks corresponding to the original image
    # bbox: cropped area
    # landmarks : the predictions of network
    
    '''rescale'''
    h = 700
    w = 700
    new_h = 700
    new_w = 700
    # new_h, new_w = int(new_h), int(new_w)
    temp = landmarks * [new_w / w, new_h / h]
    
    '''recursion to original points'''
    landmark_re=temp-[new_h/2, new_w/2]
    center=np.array([bbox[0]+bbox[2],bbox[1]+bbox[3]])*0.5
    landmark_re+=center
    
    return center, landmark_re
    
if __name__ == "__main__":       
    

    data_transform = transforms.Compose([
        HandCrop(),
        # Resize(256),
        Rescale_padding(700),
        Normalize()
        # ToTensor()
    ])
    
    data_transform_show = transforms.Compose([
        HandCrop(),
        # Resize(256),
        Rescale_padding(700),
        # Normalize()
        # ToTensor()
    ])
    
    
    # load image to test
    
    path_image='/home/jekim/workspace/jinju_ex/data/GX010166/images'
    path_openpose="/home/jekim/workspace/jinju_ex/data/GX010166/openpose"
    path_annos='/home/jekim/workspace/jinju_ex/data/GX010166/annos'
    path_result='/home/jekim/workspace/jinju_ex/data/GX010166/result/'
    list_=sorted(os.listdir(path_annos))
        
    if len(os.listdir(path_annos))==0:
        list_=sorted(os.listdir(path_annos))
    else:
        list_ori=sorted(os.listdir(path_annos))
        list_=list_ori[len(os.listdir(path_result)):]
        
    for file_name_temp in list_:
        file_name=file_name_temp.split('.')[0][:6]
        # file_name='026300'
               
        # load image
        name_img = os.path.join(path_image,file_name+'.jpg')     
        image = io.imread(name_img)
        
        # load bbox
        name_annos = os.path.join(path_annos,file_name+'.json')
        with open(name_annos) as f: annos = json.load(f)
        
        '''left'''
        
        landmarks_bbox_l=np.array(annos['bbox_left'])
        
        bbox= landmarks_bbox_l.astype(int)
        sample = {'image': image, 'bbox':bbox}
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
        '''validation'''
    
        start_time = time.time()
        
        num_class = 6
        
        image_shape = 700
        
        with torch.no_grad():
        
            best_network = Network()
            best_network.cuda()
            best_network.load_state_dict(torch.load('/home/jekim/workspace/jinju_ex/log/20220705_152640_df2op/df2op_50.pth')) 
            best_network.eval()
            
            #transform / make to tensor
            images_temp = data_transform(sample)
            images = images_temp['image'].float().cuda() # why does it needs a float()
            images=images.unsqueeze(0)
            
            # image to show
            images_show=data_transform_show(sample)
                
            predictions = (best_network(images).cpu() + 0.5) * image_shape
            temp=predictions.detach().numpy()
            display_result=(temp[0,:].reshape(-1,2)+0.5)
            display_left=display_result
            
        # plt.figure(figsize=(10,10))
        # plt.imshow(images_show['image'])
        # plt.scatter(display_left[:,0], display_left[:,1], c = 'y', s = 50)
        # plt.xlim(0,700)
        # plt.ylim(700,0)
        # # plt.text(100,100,"up-end: {:.0f}".format(landmarks_knife[0][2]),color='y',fontsize=20) # print visibility
        # # plt.text(100,200,"down-end: {:.0f}".format(landmarks_knife[1][2]),color='y',fontsize=20)
        # plt.show()
       
        '''right'''
        
        landmarks_bbox_r=np.array(annos['bbox_right'])
        
        bbox= landmarks_bbox_r.astype(int)
        sample = {'image': image, 'bbox':bbox}
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
        '''validation'''
    
        start_time = time.time()
        
        num_class = 6
        
        image_shape = 700
        
        with torch.no_grad():
        
            best_network = Network()
            best_network.cuda()
            best_network.load_state_dict(torch.load('/home/jekim/workspace/jinju_ex/log/20220705_150238_df2op/df2op_50.pth')) 
            best_network.eval()
            
            #transform / make to tensor
            images_temp = data_transform(sample)
            images = images_temp['image'].float().cuda() # why does it needs a float()
            images=images.unsqueeze(0)
            
            # image to show
            images_show=data_transform_show(sample)
                
            predictions = (best_network(images).cpu() + 0.5) * image_shape
            temp=predictions.detach().numpy()
            display_result=(temp[0,:].reshape(-1,2)+0.5)
            display_right=display_result
            
            # plt.figure(figsize=(10,10))
            # plt.imshow(images_show['image'])
            # plt.scatter(display_right[:,0], display_right[:,1], c = 'y', s = 50)
            # plt.xlim(0,700)
            # plt.ylim(700,0)
            # # plt.text(100,100,"up-end: {:.0f}".format(landmarks_knife[0][2]),color='y',fontsize=20) # print visibility
            # # plt.text(100,200,"down-end: {:.0f}".format(landmarks_knife[1][2]),color='y',fontsize=20)
            # plt.show()
                
            
            # return to original image
            
        center_l, landmark_re_left=landmark_Recursion(landmarks_bbox_l,display_left)
        center_r, landmark_re_right=landmark_Recursion(landmarks_bbox_r,display_right)
        
        '''display'''
        
        plt.figure(figsize=(15,15*image.shape[0]/image.shape[1]))
        
        ax = plt.gca()
        ax.add_patch(
              patches.Rectangle(
                (center_l[0]-350, center_l[1]-350),
                700,
                700,
                linewidth=3,
                edgecolor = 'red',
                # facecolor = 'red',
                fill=False
              ) )
        
        ax.add_patch(
              patches.Rectangle(
                (center_r[0]-350, center_r[1]-350),
                700,
                700,
                linewidth=3,
                edgecolor = 'red',
                # facecolor = 'red',
                fill=False
              ) )
        
        
        plt.imshow(image)
        plt.scatter(landmark_re_left[:,0], landmark_re_left[:,1], c = 'y', s = 50)
        plt.scatter(landmark_re_right[:,0], landmark_re_right[:,1], c = 'b', s = 50)
        plt.xlim(0,image.shape[1])
        plt.ylim(image.shape[0],0)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.show()
    
        plt.savefig(os.path.join(path_result,file_name+'_pre.png'))
            
            
            
            
            
            
