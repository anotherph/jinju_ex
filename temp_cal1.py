#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:27:41 2022

@author: jekim
"""

import numpy as np 
import math as ma
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def find_point_on_normalvector(P1,Q1,R1,C1):
# P,Q,R : three points on the plane, C1: the point on the normalvector of plane, determine the slope
# C1 : point on the normalvector, determine the one point
# output: C1_: the point on the normalvector of plane far from C1 with distance 1 

    P1_=P1-C1
    Q1_=Q1-C1
    R1_=R1-C1
    C1_=np.array([0,0,0])
    
    # normal vector
    A1=Q1_-P1_
    B1=R1_-P1_
    nor1=np.array([A1[1]*B1[2]-A1[2]*B1[1],A1[2]*B1[0]-A1[0]*B1[2],A1[0]*B1[1]-A1[1]*B1[0]])
    nor1=nor1/np.linalg.norm(nor1)

    # distance between plane and point
    d=1

    # final postiion    
    C1_=[d*nor1[0],d*nor1[1],d*nor1[2]]
    C1_=C1_/np.linalg.norm(C1_)
    C1_f=C1_+C1
    
    # plt.title("1")
    # ax.scatter([P1_[0],Q1_[0],R1_[0]],[P1_[1],Q1_[1],R1_[1]],[P1_[2],Q1_[2],R1_[2]],c='r', marker='.', s=15)
    # ax.plot([P1_[0],Q1_[0]],[P1_[1],Q1_[1]],[P1_[2],Q1_[2]],c='r')
    # ax.plot([P1_[0],R1_[0]],[P1_[1],R1_[1]],[P1_[2],R1_[2]],c='r')
    # ax.plot([Q1_[0],R1_[0]],[Q1_[1],R1_[1]],[Q1_[2],R1_[2]],c='r')
    # ax.scatter(0,0,0,c='g',marker='^')
    # ax.scatter(C1_[0],C1_[1],C1_[2],c='g',marker='^')
    # ax.plot([C1_[0],0],[C1_[1],0],[C1_[2],0],c='g')

    # ax.set_xlabel('X-axis', fontweight ='bold')
    # ax.set_ylabel('Y-axis', fontweight ='bold')
    # # ax.set_zlabel('Z-axis', fontweight ='bold')
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])
    # # ax.set_xlim([-2,2])
    # # ax.set_ylim([-2,2])
    # # ax.set_zlim([-2,2])
    # ax.view_init(20,80)
    
    # plt.title("2")
    # ax.scatter([P1[0],Q1[0],R1[0]],[P1[1],Q1[1],R1[1]],[P1[2],Q1[2],R1[2]],c='r', marker='.', s=15)
    # ax.plot([P1[0],Q1[0]],[P1[1],Q1[1]],[P1[2],Q1[2]],c='r')
    # ax.plot([P1[0],R1[0]],[P1[1],R1[1]],[P1[2],R1[2]],c='r')
    # ax.plot([Q1[0],R1[0]],[Q1[1],R1[1]],[Q1[2],R1[2]],c='r')
    # ax.scatter(C1[0],C1[1],C1[2],c='g',marker='^')
    # ax.scatter(C1_f[0],C1_f[1],C1_f[2],c='g',marker='^')
    # ax.plot([C1[0],C1_f[0]],[C1[1],C1_f[1]],[C1[2],C1_f[2]],c='g')

    # ax.set_xlabel('X-axis', fontweight ='bold')
    # ax.set_ylabel('Y-axis', fontweight ='bold')
    # # ax.set_zlabel('Z-axis', fontweight ='bold')
    # ax.set_xlim([4,6])
    # ax.set_ylim([2,4])
    # ax.set_zlim([-1,1])
    # # ax.set_xlim([-2,2])
    # # ax.set_ylim([-2,2])
    # # ax.set_zlim([-2,2])
    # ax.view_init(20,80)
    
    return C1_f

fig = plt.figure(figsize = (7, 7))
ax = plt.axes(projection ="3d")
fig.canvas.draw()
my_cmap = plt.get_cmap('hsv')
            
# compute normal vector of plane
# P1=np.array([x1[0],y1[0],z1[0]])
# Q1=np.array([x1[1],y1[1],z1[1]])
# R1=np.array([x1[2],y1[2],z1[2]])
# P2=np.array([x2[0],y2[0],z2[0]])
# Q2=np.array([x2[1],y2[1],z2[1]])
# R2=np.array([x2[2],y2[2],z2[2]])

P1=np.array([1,1,1])
Q1=np.array([2,2,0])
R1=np.array([0,2,0])

# P1=np.array([0.56,0.29,-0.018]) # front 
# Q1=np.array([0.58,0.34,-0.016])
# R1=np.array([0.59,0.28,-0.11])

# P2=np.array([0.57,0.03,-0.005]) # up 
# Q2=np.array([0.58,0.054,-0.008])
# R2=np.array([0.59,0.035,0.0009])

# P1=P1*10
# Q1=Q1*10
# R1=R1*10

# Center of P1,Q1,R1
C1=(P1+Q1+R1)/3

C1_new=find_point_on_normalvector(P1,Q1,R1,C1)

C1_new1=find_point_on_normalvector(C1,C1_new,Q1,C1)

X_new=C1_new1-C1 
Y_new=Q1-C1
Z_new=C1_new-C1

# new coordinate
print('new coordinate:', X_new,Y_new,Z_new) #x,y,z
print('center:',C1)


# euler angle
alpha=ma.acos(round(-Z_new[1]/ma.sqrt(1-pow(Z_new[2],2))))*180/ma.pi
# alpha=ma.acos(-Z_new[1]/ma.sqrt(1-pow(Z_new[2],2)))
beta= ma.acos(Z_new[2])*180/ma.pi
gamma=ma.acos(Y_new[2]/ma.sqrt(1-pow(Z_new[2],2)))*180/ma.pi

ax.scatter([P1[0],Q1[0],R1[0]],[P1[1],Q1[1],R1[1]],[P1[2],Q1[2],R1[2]], c='r', marker='o', s=15)
ax.plot([P1[0],Q1[0]],[P1[1],Q1[1]],[P1[2],Q1[2]],c='r')
ax.plot([P1[0],R1[0]],[P1[1],R1[1]],[P1[2],R1[2]],c='r')
ax.plot([Q1[0],R1[0]],[Q1[1],R1[1]],[Q1[2],R1[2]],c='r')

ax.scatter(C1_new[0],C1_new[1],C1_new[2],c='r',marker='^')
ax.scatter(C1[0],C1[1],C1[2],c='r',marker='^')
ax.plot([C1_new[0],C1[0]],[C1_new[1],C1[1]],[C1_new[2],C1[2]],c='r')

ax.scatter([C1[0],C1_new[0],Q1[0]],[C1[1],C1_new[1],Q1[1]],[C1[2],C1_new[2],Q1[2]], c='g', marker='o', s=15)
ax.plot([C1[0],C1_new[0]],[C1[1],C1_new[1]],[C1[2],C1_new[2]],c='g')
ax.plot([C1[0],Q1[0]],[C1[1],Q1[1]],[C1[2],Q1[2]],c='g')
ax.plot([Q1[0],C1_new[0]],[Q1[1],C1_new[1]],[Q1[2],C1_new[2]],c='g')

ax.scatter(C1_new1[0],C1_new1[1],C1_new1[2],c='g',marker='^')
ax.plot([C1_new1[0],C1[0]],[C1_new1[1],C1[1]],[C1_new1[2],C1[2]],c='g')

plt.title("simple 3D scatter plot")
ax.set_xlabel('X-axis', fontweight ='bold')
ax.set_ylabel('Y-axis', fontweight ='bold')
# ax.set_zlabel('Z-axis', fontweight ='bold')
# ax.set_xlim([4,6])
# ax.set_ylim([2,4])
# ax.set_zlim([-1,1])
# ax.set_xlim([-2,2])
# ax.set_ylim([-2,2])
# ax.set_zlim([-2,2])
ax.view_init(20,100)

# ax.scatter([P1[0],Q1[0],R1[0]],[P1[1],Q1[1],R1[1]],[P1[2],Q1[2],R1[2]], c='r', marker='o', s=15)
# ax.plot([P1[0],Q1[0]],[P1[1],Q1[1]],[P1[2],Q1[2]],c='r')
# ax.plot([P1[0],R1[0]],[P1[1],R1[1]],[P1[2],R1[2]],c='r')
# ax.plot([Q1[0],R1[0]],[Q1[1],R1[1]],[Q1[2],R1[2]],c='r')

# ax.scatter([P2[0],Q2[0],R2[0]],[P2[1],Q2[1],R2[1]],[P2[2],Q2[2],R2[2]], c='g', marker='o', s=15)
# ax.plot([P2[0],Q2[0]],[P2[1],Q2[1]],[P2[2],Q2[2]],c='g')
# ax.plot([P2[0],R2[0]],[P2[1],R2[1]],[P2[2],R2[2]],c='g')
# ax.plot([Q2[0],R2[0]],[Q2[1],R2[1]],[Q2[2],R2[2]],c='g')

# plt.title("simple 3D scatter plot")
# ax.set_xlabel('X-axis', fontweight ='bold')
# ax.set_ylabel('Y-axis', fontweight ='bold')
# # ax.set_zlabel('Z-axis', fontweight ='bold')
# # ax.set_xlim([4,6])
# # ax.set_ylim([2,4])
# # ax.set_zlim([-1,1])
# # ax.set_xlim([-2,2])
# # ax.set_ylim([-2,2])
# # ax.set_zlim([-2,2])
# ax.view_init(20,50)


fig.canvas.draw()
