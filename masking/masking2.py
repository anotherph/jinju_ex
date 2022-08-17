#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:09:04 2022

we know the size of knife in image

1.load the mask image
2.load the image we want to compare
3.match hand position and bottle of knife
4.calculate the inner product of each pixel
5.rotate the mask image and repeat step4

% find the mask of knife... 

@author: jekim
"""

import os
import sys 
import cv2 
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from skimage import io, transform
import numpy.linalg as LA

def find_knife_(path_image, hand):
    
    similarity=[]
    image_ori=cv2.imread(path_image)
    # image_x=cv2.imread('/home/jekim/workspace/jinju_ex/data/1.jpg')
           
    # for ind, phi_ in enumerate(range(-50,130,10)):
    for ind, phi_ in enumerate(range(-50,130,10)):
        # phi_=210
        phi=phi_*math.pi/180
        mask_hand=(40,100) # the position of hand in mask image
        mask_knife=(195,10) # the position of the end of knife
        mask=cv2.resize(image_mask[600:2000,136:,:],(200,200))
        mask_rotate=rotate_image(mask,phi)
        
        start_time = time.time()
        mask_h=(mask_hand[0]-mask.shape[0]/2,mask.shape[1]/2-mask_hand[1])
        # mask_k=(mask_knife[0]-mask.shape[0]/2,mask.shape[1]/2-mask_knife[1])
        
        mask_h_r=(mask_h[0]*math.cos(phi)-mask_h[1]*math.sin(phi),mask_h[0]*math.sin(phi)+mask_h[1]*math.cos(phi))
        # mask_k_r=(mask_k[0]*math.cos(phi)-mask_k[1]*math.sin(phi),mask_k[0]*math.sin(phi)+mask_k[1]*math.cos(phi))
        
        mask_h_r=(mask_h_r[0]+mask_rotate.shape[0]/2,mask_rotate.shape[1]/2-mask_h_r[1])
        # mask_k_r=(mask_k_r[0]+mask_rotate.shape[0]/2,mask_rotate.shape[1]/2-mask_k_r[1])
        
               
        #check the simility without black area
        
        # left=image
        # left[int(hand_left[1])-int(mask_h_r[1]):int(hand_left[1])-int(mask_h_r[1])+int(mask_rotate.shape[1]),int(hand_left[0])-int(mask_h_r[0]):int(hand_left[0])-int(mask_h_r[0])+int(mask_rotate.shape[0])]=mask_rotate

        x=int(mask_h_r[0])
        y=int(mask_h_r[1])
        a=int(hand[0])
        b=int(hand[1])
        w=mask_rotate.shape[1]
        h=mask_rotate.shape[0]
        a1=a-x
        b1=b-y
        a2=a+(w-x)
        b2=b+(h-y)
        if a1<0:
            a1=0
        elif b1<0:
            b1=0
        arr_not0=find_not0([a1,b1],mask_rotate)
        img1=image_ori
        image_ori=cv2.imread(path_image)
        # print(ind)
        # print(a1,b1,a2,b2)
        # print(img1[b1:b2,a1:a2].shape)
        # print(mask_rotate[b1:,a1:].shape)
        img1[b1:b2,a1:a2]=mask_rotate[h-(b2-b1):h,w-(a2-a1):w]
        arr_=np.array(arr_not0)
        
        error=0
        num_error=0
        for ind_ in arr_:
            x_=ind_[0]
            y_=ind_[1]
            vec1=np.array(image_ori[y_,x_],dtype=np.int64)
            vec2=np.array(img1[y_,x_],dtype=np.int64)
            
            if math.isnan(np.inner(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2))):
                pass
            else:
                error+=np.inner(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2))
                num_error+=1
        error=error/num_error
        print(error)
        print("---{:.4f}s seconds---".format(time.time()-start_time))

        similarity.append([phi_,error])
        
    simi=np.array(similarity)
    phi_max=simi[np.argmax(simi[:,1]),0]
    
    # plt.plot(simi[:,0],simi[:,1])
    
    # phi_max=250
    phi=phi_max*math.pi/180
    mask_hand=(40,100) # the position of hand in mask image
    mask_knife=(195,10) # the position of the end of knife
    mask=cv2.resize(image_mask[600:2000,136:,:],(200,200))
    mask_rotate=rotate_image(mask,phi)
    
    mask_h=(mask_hand[0]-mask.shape[0]/2,mask.shape[1]/2-mask_hand[1])
    mask_k=(mask_knife[0]-mask.shape[0]/2,mask.shape[1]/2-mask_knife[1])
    
    mask_h_r=(mask_h[0]*math.cos(phi)-mask_h[1]*math.sin(phi),mask_h[0]*math.sin(phi)+mask_h[1]*math.cos(phi))
    mask_k_r=(mask_k[0]*math.cos(phi)-mask_k[1]*math.sin(phi),mask_k[0]*math.sin(phi)+mask_k[1]*math.cos(phi))
    
    mask_h_r=(mask_h_r[0]+mask_rotate.shape[0]/2,mask_rotate.shape[1]/2-mask_h_r[1])
    mask_k_r=(mask_k_r[0]+mask_rotate.shape[0]/2,mask_rotate.shape[1]/2-mask_k_r[1])
    
    del_x=mask_k_r[0]-mask_h_r[0]
    del_y=mask_k_r[1]-mask_h_r[1]
    
    k_x=hand[0]+del_x
    k_y=hand[1]+del_y
    
    return (k_x,k_y)

def find_knife(path_annos, path_image):
    
    image=cv2.imread(path_image)
    # image_ori=image
    with open(path_annos) as f: annos = json.load(f)
    
    landmarks_pose=np.array(annos['people'][0]['pose_keypoints_2d']).reshape(-1,3)
    
    hand_left=landmarks_pose[4,:2]
    hand_right=landmarks_pose[7,:2]
    
    k_left=find_knife_(path_image,hand_left)
    k_right=find_knife_(path_image,hand_right)
    
    return k_left, k_right
    

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    angle=angle/math.pi*180
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def find_not0(st, image):
    arr=[]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if not np.array_equal(image[x,y,:],[0,0,0]):
                arr.append([y+st[0],x+st[1]]) 
                # plt.imshow(image)
                # plt.scatter(x,y)
            else:
                pass       
    return arr

if __name__ == "__main__":
    
    '''load mask'''
    path_mask='/home/jekim/workspace/jinju_ex/masking/mask3.JPEG'
    image_mask=cv2.imread(path_mask)
    
    path_main='/home/jekim/workspace/jinju_ex/data/0720_SGU/original_video/zed_camera/geommu4/'
    path_dir=os.path.join(path_main,'images/')
    list_=sorted(os.listdir(path_dir))
    
    for file_name_temp in list_:
        file_name=file_name_temp.split('.')[0][:6]
        # file_name='00087'
        path_image=os.path.join(path_main,'images/'+file_name+'.jpg')
        path_annos=os.path.join(path_main,'openpose/'+file_name+'_keypoints.json')
        
        image = io.imread(path_image)

        '''infer the position of end of the knife'''
        
        l, r = find_knife(path_annos, path_image)
        
        with open(path_annos) as f: annos = json.load(f)
        landmarks_pose=np.array(annos['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        
        ''' save the image '''
        plt.figure(figsize=(15,15*1080/1920))
        plt.imshow(image)
        plt.scatter(l[0],l[1],c='r',s=100)
        plt.scatter(landmarks_pose[4,0],landmarks_pose[4,1],c='r',s=50)
        plt.plot((landmarks_pose[4,0],l[0]),(landmarks_pose[4,1],l[1]),c='r')
        
        plt.scatter(r[0],r[1],c='b',s=100)
        plt.scatter(landmarks_pose[7,0],landmarks_pose[7,1],c='b',s=50)
        plt.plot((landmarks_pose[7,0],r[0]),(landmarks_pose[7,1],r[1]),c='b')
        
        
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.imshow(image)
        plt.savefig(os.path.join(path_main,'result_mask/'+file_name+'.png'))
        
        ''' save the infer '''
        
        annos={}
        annos['landmarks']=(l,r)
        annos['pose']=landmarks_pose[(4,7),:2].tolist()
        
        with open(os.path.join(path_main,'result_mask/'+file_name+'_result.json'), "w") as json_file:
            json.dump(annos, json_file)

    
    
    
    



