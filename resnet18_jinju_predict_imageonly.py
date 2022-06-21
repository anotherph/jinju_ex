#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

resnet18 to detect the keypoints of knife

prediction (image only)


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

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import json
from pycocotools.coco import COCO 
import pylab

# %matplotlib inline

    
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
        image = sample[:,:,:3]
        
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
        max_wh = self.output_size+50 #padding to extended box
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
    def __init__(self,num_classes=(4)*2):
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
    
    
if __name__ == "__main__":       
    

    data_transform = transforms.Compose([
        # BodyCrop(),
        # Resize(256),
        Rescale_padding(500),
        Normalize()
        # ToTensor()
    ])
    
    data_transform_show = transforms.Compose([
        # BodyCrop(),
        # Resize(256),
        Rescale_padding(500),
        # Normalize()
        # ToTensor()
    ])
    

    
    # load image to test
    
    path_image='/home/jekim/workspace/jinju_ex/data/white1/images'
    list_=sorted(os.listdir(path_image))
        
    for file_name_temp in list_:
        file_name=file_name_temp.split('.')[0][:5]
        
        name_img = os.path.join(path_image,file_name+'.jpg')     
    
        # name_img = "/home/jekim/workspace/jinju_ex/data/1.png"        # name_img="/home/jekim/workspace/jinju_ex/data/white2/images/00152.jpg"
        image = io.imread(name_img)
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    
        '''validation'''
    
        start_time = time.time()
        
        num_class = 4
        
        image_shape = 550
        
        with torch.no_grad():
        
            best_network = Network()
            best_network.cuda()
            best_network.load_state_dict(torch.load('/home/jekim/workspace/jinju_ex/log/20220607_112922_df2op/df2op_25.pth')) 
            best_network.eval()
            
            #transform / make to tensor
            images_temp = data_transform(image)
            images = images_temp['image'].float().cuda() # why does it needs a float()
            images=images.unsqueeze(0)
            
            # image to show
            images_show=data_transform_show(image)
                
            predictions = (best_network(images).cpu() + 0.5) * image_shape
            temp=predictions.detach().numpy()
            display_result=(temp[0,:].reshape(-1,2)+0.5)
            # predictions = predictions.view(-1,num_class,2) # what does this mean?
            
            # plt.figure(figsize=(10,40))
            plt.figure(figsize=(10,10))
            
            # for img_num in range(1):
                # plt.subplot(8,1,img_num+1)
            plt.imshow(images_show['image'])
            plt.scatter(display_result[:,0], display_result[:,1], c = 'r', s = 15)
            plt.xlim(0,image_shape)
            plt.ylim(image_shape,0)
                # plt.scatter(predictions[0,:], predictions[1,:], c = 'r', s = 5)
            
            plt.savefig(os.path.join('/home/jekim/workspace/jinju_ex/data/white1/result/',file_name+'_pre.png'))
