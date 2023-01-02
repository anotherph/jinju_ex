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

# Center of P1,Q1,R1 and P2,Q2,R2
C1=(P1+Q1+R1)/3

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
C1_new=[d*nor1[0]+C1[0],d*nor1[1]+C1[1],d*nor1[2]+C1[2]]

C1_=[d*nor1[0],d*nor1[1],d*nor1[2]]
C1_=C1_/np.linalg.norm(C1_)



theta=ma.atan2(C1_[2],ma.sqrt(pow(C1_[0],2)+pow(C1_[1],2)))*180/ma.pi
phi=ma.atan2(C1_[1],C1_[0])*180/ma.pi


ax.scatter([P1[0],Q1[0],R1[0]],[P1[1],Q1[1],R1[1]],[P1[2],Q1[2],R1[2]], c='r', marker='o', s=15)
ax.plot([P1[0],Q1[0]],[P1[1],Q1[1]],[P1[2],Q1[2]],c='r')
ax.plot([P1[0],R1[0]],[P1[1],R1[1]],[P1[2],R1[2]],c='r')
ax.plot([Q1[0],R1[0]],[Q1[1],R1[1]],[Q1[2],R1[2]],c='r')

ax.scatter([P1_[0],Q1_[0],R1_[0]],[P1_[1],Q1_[1],R1_[1]],[P1_[2],Q1_[2],R1_[2]], c='c', marker='o', s=15)
ax.plot([P1_[0],Q1_[0]],[P1_[1],Q1_[1]],[P1_[2],Q1_[2]],c='c')
ax.plot([P1_[0],R1_[0]],[P1_[1],R1_[1]],[P1_[2],R1_[2]],c='c')
ax.plot([Q1_[0],R1_[0]],[Q1_[1],R1_[1]],[Q1_[2],R1_[2]],c='c')

ax.scatter(C1_new[0],C1_new[1],C1_new[2],c='b',marker='^')
ax.scatter(C1[0],C1[1],C1[2],c='b',marker='^')
ax.plot([C1_new[0],C1[0]],[C1_new[1],C1[1]],[C1_new[2],C1[2]],c='b')

ax.scatter(nor1[0],nor1[1],nor1[2],c='c',marker='^')
ax.scatter(C1_[0],C1_[1],C1_[2],c='c',marker='^')
ax.plot([nor1[0],C1_[0]],[nor1[1],C1_[1]],[nor1[2],C1_[2]],c='c')


plt.title("simple 3D scatter plot")
ax.set_xlabel('X-axis', fontweight ='bold')
ax.set_ylabel('Y-axis', fontweight ='bold')
ax.set_zlabel('Z-axis', fontweight ='bold')
ax.set_xlim([-0,2])
ax.set_ylim([-0,2])
ax.set_zlim([-0,2])
ax.view_init(20,180)

fig.canvas.draw()
