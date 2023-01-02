import re
import os
import numpy as np

'''make mp4'''
'''

'''

# path = '/media/jekim/1305f5d2-41f6-41cd-869b-f75a15043f77/home/jekim/EasyMocap-MJ1/result_collection/8_SC_1026/scene_1_1/output'
# pathOut='/media/jekim/1305f5d2-41f6-41cd-869b-f75a15043f77/home/jekim/EasyMocap-MJ1/result_collection/8_SC_1026/scene_1_1/output/result_1.mp4'

path='/home/jekim/workspace/jinju_ex/data/0720_SGU/original_video/zed_camera/geommu3/result_mask'
pathOut='/home/jekim/workspace/jinju_ex/data/0720_SGU/original_video/zed_camera/geommu3/result_mask/video_withmask.mp4'


paths = [os.path.join(path , i ) for i in os.listdir(path) if re.search(".png$", i )]
## 정렬 작업
store1 = []
store2 = []
for i in paths :
    if len(i) == 300 :
        store2.append(i)
    else :
        store1.append(i)

paths = list(np.sort(store1)) + list(np.sort(store2))

#len('ims/2/a/2a.2710.png')
#pathIn= './jekim7/output/smpl/'

fps = 30
import cv2
frame_array = []
for idx , path in enumerate(paths) : 
    # if (idx % 2 == 0) | (idx % 5 == 0) :
    #     continue
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
