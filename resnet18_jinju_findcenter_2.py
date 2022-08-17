#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resnet18 to detect the center of knife (instead of bbox regression)

1. crop the area of hands 
2. learning the cropped images to detect a center-point of knife

edit of 'class Jinju_Dataset'
use annos_split

fine the end of knife

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import json
from pycocotools.coco import COCO 
import pylab
import random

%matplotlib inline

class Jinju_Dataset(Dataset): 
    
    def __init__(self, root_image, root_annos, transform=None):

        self.root_image = root_image # directory and files 
        self.root_annos = root_annos 
        self.transform = transform
        self.num_class=1

    def __len__(self):
        return len(os.listdir(self.root_annos))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        dir_list_img=os.listdir(self.root_annos)
        file_name=os.path.splitext(dir_list_img[idx])[0]
              
        name_annos = os.path.join(self.root_annos,file_name+'.json')
        with open(name_annos) as f: annos = json.load(f)
        
        name_img = os.path.join(self.root_image,annos['name of image'])
        image = io.imread(name_img)
        
        landmarks_pose=np.array(annos['body_keypoints']).reshape(-1,3)
                
        if annos['position']=='left':
            ind_hand = 9 # left hand: 9, right hand: 10
        elif annos['position']=='right':
            ind_hand = 10 # left hand: 9, right hand: 10
        space=150
        bbox_0=int(landmarks_pose[ind_hand,0])-space
        if bbox_0<0: # small x 
            bbox_0=0
        bbox_1=int(landmarks_pose[ind_hand,1])-space
        if bbox_1<0:
            bbox_1=0
        bbox_2=int(landmarks_pose[ind_hand,0])+space
        if bbox_2>image.shape[1]: # large x 
            bbox_2=image.shape[1]
        bbox_3=int(landmarks_pose[ind_hand,1])+space
        if bbox_3>image.shape[0]:
            bbox_3=image.shape[0]
        
        bbox = [bbox_0,bbox_1,bbox_2,bbox_3]
        center=np.array(annos['points'])
        center=center[:2].reshape(1,2)
        landmark_vis = np.zeros((1,2))            
        
        sample = {'image': image, 'landmarks': center, 'bbox':bbox}
        
        # # '''plot'''
        # plt.imshow(image)
        # plt.scatter(center[:,0], center[:,1], c = 'y', s = 10)
        # # plt.scatter(landmarks_pose[ind_hand,0],landmarks_pose[ind_hand,1], c = 'k',s=10)
        # plt.scatter(landmarks_pose[:,0],landmarks_pose[:,1], c = 'm',s=10)
        # plt.scatter(bbox[0],bbox[1],c='b',s=5)
        # plt.scatter(bbox[2],bbox[3],c='r',s=5)
        
        # plt.show()

        if self.transform:
            sample = self.transform(sample)
            
        # for i, landmark in enumerate(landmarks) :
        #     if landmark[0]<0 or landmark[1]<0:
        #         landmarks[i,:]=[0, 0, 0]
        #     elif landmark[0]>300 or landmark[1]>300:
        #         landmarks[i,:]=[0, 0, 0]
    
        # if landmarks don't exist in cropped image, landmark_vis set to be 0.
        for i, center_ in enumerate(sample['landmarks']) :
            if center_[0]<0 and center_[1]<0:
                landmark_vis[i]=np.array([0,0])
            else:
                landmark_vis[i]=np.array([1,1])
                
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
#
        # print(landmarks[:].shape,landmark_vis[:].shape,file_name)
        return image, np.append(landmarks,landmark_vis.reshape(self.num_class,-1),axis=1) #return the landmarks with visible index
             
class HandCrop(object):
    " crop the area of the hand in image"
    
    def __call__(self, sample):
        image, landmarks, bbox = sample['image'], sample['landmarks'], sample['bbox']
    
        '''bbox = [x1,y1,x2,y2]'''        
        
        
        image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        landmarks = np.array(landmarks) - np.array([bbox[0], bbox[1]])
        
        '''determine the landmarks exist in cropped image'''
        
        for i, landmark in enumerate(landmarks) :
            if landmark[0]<0 or landmark[1]<0:
                landmarks[i,:]=[0, 0]
            elif landmark[0]>image.shape[1] or landmark[1]>image.shape[0]:
                landmarks[i,:]=[0, 0]
        
        plt.scatter(landmarks[:,0], landmarks[:,1], c = 'r', s = 5)
        plt.imshow(image)
        # plt.show()
        return {'image': image, 'landmarks': landmarks}

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
    def __init__(self,num_classes=(1)*2):
        super().__init__()
        self.model=models.resnet18()
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
    
    train_img_dir = "/home/jekim/workspace/jinju_ex/data/0720_SGU/dataset_knife/images"
    train_json_path ="/home/jekim/workspace/jinju_ex/data/0720_SGU/dataset_knife/annos_split"

    data_transform = transforms.Compose([
        HandCrop(),
        # Resize(256),
        Rescale_padding(300),
        Normalize()
        # ToTensor()
    ])
    
    dataset = Jinju_Dataset(root_image=train_img_dir,root_annos=train_json_path,transform=data_transform)
    # dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [661,165])
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [800,202])
    
    image, landmarks=dataset_valid[6] # check the data and length of tensor
    temp=image.cpu().detach().numpy()
    plt.imshow(temp.transpose(1,2,0))
    
    batch_size= 32
    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size,shuffle=True, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    ''' train '''
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    dir_log= os.path.join("/home/jekim/workspace/jinju_ex/log",timestr+'_df2op')
    os.makedirs(dir_log)
    
    '''load Network'''
    network = Network() 
    network.to(device)
    
    '''load Network using pretrained model'''
    # PATH=os.path.join('/home/jekim/workspace/jinju_ex/log/20220502_204134_df2op', "df2op_28.pth") # pretrained model path (cocodataset_pose keypoint)
    # network = Network()
    
    # # replace the model_dict value with the pretrained_dict value except the full-connected layer 
    # pretrained_dict = torch.load(PATH)
    # model_dict = network.state_dict()
    # for k, v in pretrained_dict.items():
    #     # if k in model_dict: # if their key names are same
    #     if model_dict[k].size()==pretrained_dict[k].size(): # if their size of torch are same
    #         model_dict[k]=v
    #     else:
    #         pass
    # network.load_state_dict(model_dict)
    # network.to(device)
    
    # criterion = nn.MSELoss()
    # criterion = network.cal_loss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(network.parameters(), lr=0.0001)
    optimizer = optim.Adam(network.parameters(), lr=1e-6)
    # optimizer = optim.Adam(network.model.fc.parameters(),lr=0.0001) # turning the last fc layer 
    # optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    
    loss_min = np.inf
    num_epochs = 50
    num_class = 1
    
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
                
                '''visuliaze to check the image and landmarks'''
                temp=images.cpu().detach().numpy()
                display_img=np.transpose(temp[0,:,:,:], (1,2,0))
                temp=landmarks.cpu().detach().numpy()
                display_landmarks=(temp[0,:].reshape(-1,2)+0.5)
                temp=predictions.cpu().detach().numpy()
                display_result=(temp[0,:].reshape(-1,2)+0.5)
    
                plt.scatter(display_landmarks[:,0]*display_img.shape[0], display_landmarks[:,1]*display_img.shape[1], c = 'r', s = 5)
                plt.scatter(display_result[:,0]*display_img.shape[0], display_result[:,1]*display_img.shape[1], c = 'b', s = 5)
    
                plt.imshow(display_img.squeeze())
                plt.show()
                ''''''
    
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
        
        # save the loss value in txt file
        with open(os.path.join(dir_log,'loss_value.txt'),'a+') as file:
            file.write('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f} \n'.format(epoch, loss_train, loss_valid))                  
        
        loss_valid_save=np.append(loss_valid_save,loss_valid)
        loss_train_save=np.append(loss_train_save,loss_train)
        
        # if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(network.state_dict(), os.path.join(dir_log,'df2op_'+str(epoch)+'.pth')) 
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
    file.close()
    


        
        
