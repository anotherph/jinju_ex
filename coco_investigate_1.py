#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:27:30 2022

@author: jekim

cocodataset load and investigate
check the ground truth of heatmap and paf vector
"""

from pycocotools.coco import COCO 
import numpy as np 
import skimage.io as io 
import matplotlib.pyplot as plt 
import pylab
import argparse
import torch
import torch.nn as nn

%matplotlib inline 

def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--loader-workers', default=8, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=72, type=int,
                       help='batch size')
    group.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    group.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)') 
    group.add_argument('--nesterov', dest='nesterov', default=True, type=bool)     
    group.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')    

def train_factory(args, preprocess, target_transforms):
    train_datas = [datasets.CocoKeypoints(
        root=args.train_image_dir,
        annFile=item,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    ) for item in args.train_annotations]

    train_data = torch.utils.data.ConcatDataset(train_datas)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    val_data = datasets.CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data
    
def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(parser)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--freeze-base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--pre-lr', type=float, default=1e-4,
                        help='pre learning rate')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=368, type=int,
                        help='square edge of input images')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--debug-without-plots', default=False, action='store_true',
                        help='enable debug but dont plot')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')                        
    parser.add_argument('--model_path', default='./network/weight/', type=str, metavar='DIR',
                    help='path to where the model saved')                         
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
        
    return args

if __name__ == "__main__":
    
    IMAGE_DIR_TRAIN='/media/jekim/Samsung_T5/COCO2017/train2017/'
    ANNOTATIONS_TRAIN='/media/jekim/Samsung_T5/COCO2017/annotations/person_keypoints_train2017.json'
    IMAGE_DIR_VAL='/media/jekim/Samsung_T5/COCO2017/val2017/'
    ANNOTATIONS_VAL='/media/jekim/Samsung_T5/COCO2017/annotations/person_keypoints_val2017.json'
    
    args = cli()
    
    print("Loading dataset...")
    # load train data
    preprocess = transforms.Compose([
            transforms.Normalize(),
            transforms.RandomApply(transforms.HFlip(), 0.5),
            transforms.RescaleRelative(),
            transforms.Crop(args.square_edge),
            transforms.CenterPad(args.square_edge),
        ])
    train_loader, val_loader, train_data, val_data = train_factory(args, preprocess, target_transforms=None)

    



