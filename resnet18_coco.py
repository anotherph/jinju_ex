#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:53:29 2022

model: resnet18
data: cocodataset(person_keypoint)

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

%matplotlib inline

class COCO_Dataset(Dataset):
    
    def __init__(self, root_image, root_annos, transform=None):

        self.root_image = root_image # directory and files 
        self.root_annos = root_annos 
        self.transform = transform
        self.num_class=17

    def __len__(self):
        return len(os.listdir(self.root_image))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        with open(os.path.join(self.root_annos,str(idx)+'.json')) as f: annos = json.load(f)
        # dir_list_img=os.listdir(self.root_image)
        file_name=annos['id']
        
        name_img=os.path.join(self.root_image,str(file_name)+'.jpg')
        image = io.imread(name_img)
        
        # if len(image.shape)!=3:
        #     print(name_img)            # grey image no..... T.T
        
        landmarks=np.array(annos['keypoints']).reshape(-1,3)
        
        for ind, landmark in enumerate(landmarks):
            if landmark[-1]==2:
                landmark[-1]=1
            else: 
                landmark[-1]=0
            landmarks[ind,:]=landmark
        
        bbox = np.array(annos['bbox'],dtype=int)
        landmark_vis = np.repeat(landmarks[:,-1],2,axis=0).reshape(-1,2)
        sample = {'image': image, 'landmarks': landmarks[:,:2], 'bbox': bbox}
        
        # '''plot'''
        # plt.imshow(image)
        # plt.scatter(landmarks[:,0], landmarks[:,1], c = 'c', s = 5)
        # plt.show()

        if self.transform:
            sample = self.transform(sample)

        # return sample
        image = sample['image']
        landmarks = sample['landmarks'] - 0.5
        
        # '''visuliaze to check the image and landmarks'''
        # temp=image.numpy()
        # display_img=np.transpose(temp, (1,2,0))
        # temp=landmarks
        # display_landmarks=(temp.reshape(-1,2)+0.5)
        # # temp=predictions.cpu().detach().numpy()
        # # display_result=(temp[0,:].reshape(-1,2)+0.5)
      
        # plt.scatter(display_landmarks[:,0]*display_img.shape[0], display_landmarks[:,1]*display_img.shape[1], c = 'r', s = 5)
        # # plt.scatter(display_result[:,0]*display_img.shape[0], display_result[:,1]*display_img.shape[1], c = 'b', s = 5)
      
        # plt.imshow(display_img.squeeze())
        # plt.show()
        # ''''''   
        
        return image, np.append(landmarks,landmark_vis.reshape(self.num_class,-1),axis=1) #return the landmarks with visible index
             
class BodyCrop(object):
    " crop the area of the cloth in image"
    
    def __call__(self, sample):
        image, landmarks, bbox = sample['image'], sample['landmarks'], sample['bbox']
    
        '''bbox = [x1,y1,width,height]'''        
        
        image = image[bbox[1] : bbox[1]+bbox[3], bbox[0]: bbox[0]+bbox[2]]        
        landmarks = landmarks - [bbox[0], bbox[1]]
        
        '''extended bbox'''
        
        # spare=30
        # if bbox[3]+spare < image.shape[0] and bbox[2]+spare < image.shape[1] and bbox[1]-spare>0 and bbox[0]-spare>0:
        #     image = image[bbox[1]-spare: bbox[3]+spare, bbox[0]-spare: bbox[2]+spare]
        #     landmarks = landmarks - [bbox[0]-spare, bbox[1]-spare]
        # else:
        #     image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        #     landmarks = landmarks - [bbox[0], bbox[1]]
    
        
    
        return {'image': image, 'landmarks': landmarks}
    
class Rescale(object):
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
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
    
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
        image, landmarks = sample['image'], sample['landmarks']
        
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
        landmarks = landmarks * [new_w / w, new_h / h]
        
        '''padding'''
        
        h, w = img.shape[:2]
        # max_wh = np.max([w, h]) #padding
        max_wh = self.output_size+30 #padding to extended box
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5 # left
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5 # top
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5 # right
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5 # bottom
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        
        img_padded = np.ones((max_wh,max_wh,3))*(-1)
        img_padded[int(b_pad):int(b_pad)+h,int(l_pad):int(l_pad)+w,:]=img
        
        landmarks = landmarks + [int(l_pad),int(b_pad)]
                

        return {'image': img_padded, 'landmarks': landmarks}

class Normalize(object):

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        '''normalize the image using mean & var'''
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5], [0.5])
            ])
        
        img_normalized = transform_norm(image)
        img=np.array(img_normalized)
        
        landmarks = landmarks / [img.transpose(1,2,0).shape[1], img.transpose(1,2,0).shape[0]]
        
        # image = TF.normalize(image, [0.5], [0.5])
        
        # return {'image': img.transpose(1,2,0), 'landmarks': landmarks}
        
        return {'image': img_normalized, 'landmarks': landmarks}
    
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
    def __init__(self,num_classes=17*2):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=True)
        # self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #in case of the number of input channel = 1 (gray scale)
        # for param in self.model.parameters():
        #     param.requires_grad = False
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
    
    train_img_dir = "/home/jekim/Documents/COCO2017/train2017"
    # train_json_path = "/media/jekim/Samsung_T5/COCO2017/annotations/person_keypoints_train2017.json"
    train_json_path ="/home/jekim/Documents/COCO2017_JK/train"
    valid_img_dir = "/home/jekim/Documents/COCO2017/val2017"
    # valid_json_path = "/media/jekim/Samsung_T5/COCO2017/annotations/person_keypoints_val2017.json"
    valid_json_path ="/home/jekim/Documents/COCO2017_JK/valid"
    data_transform = transforms.Compose([
        BodyCrop(),
        # Resize(256),
        Rescale_padding(300),
        Normalize()
        # ToTensor()
    ])
    
    dataset_train = COCO_Dataset(root_image=train_img_dir,root_annos=train_json_path,transform=data_transform)
    dataset_valid = COCO_Dataset(root_image=valid_img_dir,root_annos=valid_json_path,transform=data_transform)
        
    image, landmarks=dataset_valid[200] # check the data and length of tensor
    
    batch_size= 32 
    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size,shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    ''' train '''
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    dir_log= os.path.join("/home/jekim/workspace/jinju_ex/log",timestr+'_df2op')
    os.makedirs(dir_log)
    
    network = Network()
    network.to(device)
    
    # criterion = nn.MSELoss()
    # criterion = network.cal_loss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(network.parameters(), lr=0.0001)
    optimizer = optim.Adam(network.parameters(), lr=1e-7)
    # optimizer = optim.Adam(network.model.fc.parameters(),lr=0.0001) # turning the last fc layer 
    # optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    
    loss_min = np.inf
    num_epochs = 1
    num_class = 17
    
    start_time = time.time()
    
    loss_valid_save = np.array([])
    loss_train_save = np.array([])
    
    for epoch in range(1,num_epochs+1):
        
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        
        network.train()
        for step in range(1,len(train_loader)+1):
        
            images, landmarks = next(iter(train_loader))
            
            images = images.float().cuda()
            # images = images.cuda()

            landmarks_val = landmarks[:,:,:2].reshape(batch_size,-1).float().cuda()
            landmarks_vis = landmarks[:,:,2:].reshape(batch_size,-1).float().cuda()
        
            predictions = network(images)
            
            # '''visuliaze to check the image and landmarks'''
            # temp=images.cpu().detach().numpy()
            # display_img=np.transpose(temp[0,:,:,:], (1,2,0))
            # temp=landmarks_val.cpu().detach().numpy()
            # display_landmarks=(temp[0,:].reshape(-1,2)+0.5)
            # temp=predictions.cpu().detach().numpy()
            # display_result=(temp[0,:].reshape(-1,2)+0.5)

            # plt.scatter(display_landmarks[:,0]*display_img.shape[0], display_landmarks[:,1]*display_img.shape[1], c = 'r', s = 5)
            # plt.scatter(display_result[:,0]*display_img.shape[0], display_result[:,1]*display_img.shape[1], c = 'b', s = 5)

            # plt.imshow(display_img.squeeze())
            # plt.show()
            # ''''''
            
            # clear all the gradients before calculating them
            optimizer.zero_grad()
            
            # find the loss for the current step
            # loss_train_step = criterion(predictions, landmarks)
            loss_train_step = network.cal_loss(predictions, landmarks_val, landmarks_vis)
            
            # calculate the gradients
            loss_train_step.backward()
            
            # update the parameters
            optimizer.step()
            
            loss_train += loss_train_step.item()
            running_loss = loss_train/step
            
            print_overwrite(step, len(train_loader), running_loss, 'train')
            
        network.eval() 
        with torch.no_grad():
            
            for step in range(1,len(valid_loader)+1):
                
                images, landmarks = next(iter(valid_loader))
            
                images = images.float().cuda()
                # images = images.cuda()
                # landmarks = landmarks.view(landmarks.size(0),-1).float().cuda()
                # landmarks = landmarks.view(landmarks.size(0),-1).cuda()
                
                landmarks_val = landmarks[:,:,:2].reshape(batch_size,-1).float().cuda()
                landmarks_vis = landmarks[:,:,2:].reshape(batch_size,-1).float().cuda()
                            
                predictions = network(images)
                
                # '''visuliaze to check the image and landmarks'''
                # temp=images.cpu().detach().numpy()
                # display_img=np.transpose(temp[0,:,:,:], (1,2,0))
                # temp=landmarks.cpu().detach().numpy()
                # display_landmarks=(temp[0,:].reshape(-1,2)+0.5)
                # temp=predictions.cpu().detach().numpy()
                # display_result=(temp[0,:].reshape(-1,2)+0.5)
    
                # plt.scatter(display_landmarks[:,0]*display_img.shape[0], display_landmarks[:,1]*display_img.shape[1], c = 'r', s = 5)
                # plt.scatter(display_result[:,0]*display_img.shape[0], display_result[:,1]*display_img.shape[1], c = 'b', s = 5)
    
                # plt.imshow(display_img.squeeze())
                # plt.show()
                # ''''''
    
                # find the loss for the current step
                # loss_valid_step = criterion(predictions, landmarks)
                loss_valid_step = network.cal_loss(predictions, landmarks_val, landmarks_vis)
    
                loss_valid += loss_valid_step.item()
            
                running_loss = loss_valid/step
    
                print_overwrite(step, len(valid_loader), running_loss, 'valid')
        
        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)
        
        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')
        
        loss_valid_save=np.append(loss_valid_save,loss_valid)
        loss_train_save=np.append(loss_train_save,loss_train)
        
        # if loss_valid < loss_min:
        loss_min = loss_valid
        # torch.save(network.state_dict(), os.path.join(dir_log,'df2op_'+str(epoch)+'.pth')) # save only the parameter value 
        torch.save(network,os.path.join(dir_log,'df2op_'+str(epoch)+'.pth')) # save the architecture and parameter value 
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')
        plt.plot(range(epoch),loss_train_save,'b-o',label='train loss')
        plt.plot(range(epoch),loss_valid_save,'r-o',label='validation loss')
        # legend_without_duplicate_labels(plt)
        # plt.legend()
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel("loss function")
        # plt.show()
        plt.savefig(os.path.join(dir_log,'df2op_loss_function_'+str(epoch)+'.png'), dpi=300)
         
    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
    
    # plt.plot(range(num_epochs),loss_train_save)
    # plt.plot(range(num_epochs),loss_valid_save)
    # plt.savefig('./loss_fuction.png', dpi=300)
    # plt.grid(True)
    # plt.xlabel("epoch")
    # plt.ylabel("loss function")
    # plt.show()

        
        
