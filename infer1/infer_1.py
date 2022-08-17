#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:28:05 2022

Discription: To infer the landmarks of two knives.
To do list: check the time-lapse of infering landmarks with one-image. 

# load the pretrained network once

main code contains below: 
1) load the image
2) load the body-keypoints (we need keypoints of both hands, left and right)
3) call the function to predict the center of knife
    input: hand-keypoint, original image
    output : the center of knife corresponding to the original image
4) call the function to predict the landmarks
    input: the center of knife, original image
    output: the landmarks of knife corresponding to the original image
    
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
    
class CenterCrop(object):
    " crop image along bbox of which center is sample['center'] "
    
    def __call__(self, sample):
        image, position = sample['image'], sample['center']
        # image = sample[:,:,:3]
    
        '''bbox = [x1,y1,x2,y2]'''        
        space=200
        bbox_0=int(position[0])-space
        bbox_1=int(position[1])-space
        bbox_2=int(position[0])+space
        bbox_3=int(position[1])+space
        bbox = [bbox_0,bbox_1,bbox_2,bbox_3]
        
        image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]       
            
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
    #def __init__(self,num_classes=(2)*2):
    def __init__(self, num):
        super().__init__()
        self.num_class = num*2
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc=nn.Linear(self.model.fc.in_features, self.num_class)
        
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
    
def landmark_Recursion(pose, landmarks):
    # recursion of landmarks corresponding to the original image
    # pose : the center of cropped image
    # landmarks : the predictions of network
    
    '''rescale'''
    h = 300
    w = 300
    new_h = 400
    new_w = 400
    # new_h, new_w = int(new_h), int(new_w)
    temp = landmarks * [new_w / w, new_h / h]
    
    '''recursion to original points'''
    landmark_re=temp-[new_h/2, new_w/2]
    landmark_re+=pose[:2]
    
    return landmark_re

def infer_func(sample):
    
    start_time = time.time()
    
    '''left'''
    
    ind=9
    
    '''infer center of knife'''
    position=sample['pose'][ind,:2]
    sample_center = {'image': image, 'center':position} # center = hand postion
    # net_center='/home/jekim/workspace/jinju_ex/infer1/model/center_left.pth'
    center_position_left=infer_center(1, sample_center, num_class=1)
    
    '''infer landmark of knife'''
    sample_landmark = {'image': image, 'center':center_position_left} # center = the center of knife 
    # net_landmark='/home/jekim/workspace/jinju_ex/infer1/model/landmark_left.pth'
    landmarks_left=infer_landmarks(2, sample_landmark, num_class=2)
               
    '''right'''
    
    ind=10
    
    '''infer center of knife'''
    position=sample['pose'][ind,:2]
    sample_center = {'image': image, 'center':position} # center = hand postion
    # net_center='/home/jekim/workspace/jinju_ex/infer1/model/center_right.pth'
    center_position_right=infer_center(3, sample_center, num_class=1)
    
    '''infer landmark of knife'''
    sample_landmark = {'image': image, 'center':center_position_right} # center = the center of knife 
    # net_landmark='/home/jekim/workspace/jinju_ex/infer1/model/landmark_right.pth'
    landmarks_right=infer_landmarks(4, sample_landmark, num_class=2)
    
    landmarks=np.append(landmarks_left,landmarks_right,axis=0)
    center_position=np.append(center_position_left,center_position_right,axis=0)
    
    print("---{:.4f}s seconds---".format(time.time()-start_time))
    
    return landmarks, center_position.reshape(2,2)

def infer_center(net, sample, num_class):
    
    '''define the transform the data'''
    
    data_transform = transforms.Compose([
        CenterCrop(),
        Rescale_padding(300),
        Normalize()
    ])
    
    '''validation'''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
        
    image_shape = 300
    
    # with torch.no_grad():

    #     best_network = Network(num_class)
    #     best_network.cuda()
    #     best_network.load_state_dict(torch.load(net)) 
    #     best_network.eval()
        
    #transform / make to tensor
    images_temp = data_transform(sample)
    images = images_temp['image'].float().cuda() # why does it needs a float()
    images=images.unsqueeze(0)
    
    if net==1:
        predictions = (best_network1(images).cpu() + 0.5) * image_shape
    elif net==3:
        predictions = (best_network3(images).cpu() + 0.5) * image_shape
        
    temp=predictions.detach().numpy()
    display_result=(temp[0,:].reshape(-1,2)+0.5)
    landmark_re=landmark_Recursion(sample['center'],display_result)
    
    return landmark_re[0]

def infer_landmarks(net, sample, num_class):
    
    '''define the transform the data'''
    
    data_transform = transforms.Compose([
        CenterCrop(),
        Rescale_padding(300),
        Normalize()
    ])
    
    '''validation'''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()
        
    image_shape = 300
    
    # with torch.no_grad():
    
    #     best_network = Network(num_class)
    #     best_network.cuda()
    #     best_network.load_state_dict(torch.load(net)) 
    #     best_network.eval()
        
    #transform / make to tensor
    images_temp = data_transform(sample)
    images = images_temp['image'].float().cuda() # why does it needs a float()
    images=images.unsqueeze(0)
    
    if net==2:
        predictions = (best_network2(images).cpu() + 0.5) * image_shape
    elif net==4:
        predictions = (best_network4(images).cpu() + 0.5) * image_shape
    # predictions = (best_network(images).cpu() + 0.5) * image_shape
    temp=predictions.detach().numpy()
    display_result=(temp[0,:].reshape(-1,2)+0.5)
    landmark_re=landmark_Recursion(sample['center'],display_result)
    
    return landmark_re


if __name__ == "__main__":       

    
    path_image='/home/jekim/workspace/jinju_ex/data/white1/images'
    path_annos_tot="/home/jekim/workspace/jinju_ex/data/white1/annos_tot2"
    list_=sorted(os.listdir(path_image))
    
    ''' load pretrained network'''
    with torch.no_grad():
        net1='/home/jekim/workspace/jinju_ex/infer1/model/center_left.pth'
        best_network1 = Network(1)
        best_network1.cuda()
        best_network1.load_state_dict(torch.load(net1)) 
        best_network1.eval()
        print(best_network1)
    
    with torch.no_grad():
        net2='/home/jekim/workspace/jinju_ex/infer1/model/landmark_left.pth'
        best_network2 = Network(2)
        best_network2.cuda()
        best_network2.load_state_dict(torch.load(net2)) 
        best_network2.eval()
    
    with torch.no_grad():
        net3='/home/jekim/workspace/jinju_ex/infer1/model/center_right.pth'
        best_network3 = Network(1)
        best_network3.cuda()
        best_network3.load_state_dict(torch.load(net3)) 
        best_network3.eval()
    
    with torch.no_grad():
        net4='/home/jekim/workspace/jinju_ex/infer1/model/landmark_right.pth'
        best_network4 = Network(2)
        best_network4.cuda()
        best_network4.load_state_dict(torch.load(net4)) 
        best_network4.eval()
    
    for file_name_temp in list_:
        file_name=file_name_temp.split('.')[0][:5]
        
        # file_name='00396'
        # load image
        name_img = os.path.join(path_image,file_name+'.jpg')     
        image = io.imread(name_img)
        
        # load pose_keypoints
        name_annos = os.path.join(path_annos_tot,file_name+'.json')
        with open(name_annos) as f: annos = json.load(f)
               
        landmarks_pose=np.array(annos['pose_keypoints_2d']).reshape(-1,3)
        
        if landmarks_pose[9,:].tolist()==[0,0,0] or landmarks_pose[10,:].tolist()==[0,0,0]:     
            landmarks=np.zeros((4,2))
            center=np.zeros((2,2))            
        else: 
            sample = {'image': image ,'pose':landmarks_pose}
            landmarks, center=infer_func(sample)
     
        '''display and save the result'''
            
        # fig, ax = plt.subplots()
        plt.figure(figsize=(15,15*1080/1920))
        ax = plt.gca()
        ax.add_patch(
              patches.Rectangle(
                (center[0,0]-200, center[0,1]-200),
                400,
                400,
                edgecolor = 'red',
                # facecolor = 'red',
                fill=False
              ) )
        
        ax.add_patch(
              patches.Rectangle(
                (center[1,0]-200, center[1,1]-200),
                400,
                400,
                edgecolor = 'red',
                # facecolor = 'red',
                fill=False
              ) )
        
        plt.scatter(landmarks[:,0], landmarks[:,1], c = 'y', s = 50)
        plt.scatter(center[:,0], center[:,1], c = 'r', s = 50)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.imshow(image)
        # plt.show()
        # plt.savefig(os.path.join('/home/jekim/workspace/jinju_ex/infer1/result1/',file_name+'.png'))
        
        
        
        
    

