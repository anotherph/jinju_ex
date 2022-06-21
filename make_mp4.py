import re
import os
import numpy as np

# os.system(f'ffmpeg -r 30 -i jekim7/output/smpl.*.png'
#               f' -crf 30 jekim7/output/smpl.mp4')
              
# path = "./4_exm1/output/smpl/"
# pathOut = './result_onlyvideo/SGU_1001/jekim_90deg_smplh.mp4'

# path = '/home/jekim/workspace/EasyMocap-MJ1/etc/temp/'
# pathOut ='/home/jekim/workspace/EasyMocap-MJ1/etc/sorigun.mp4'

path = '/home/jekim/workspace/jinju_ex/data/white1/result_all'
pathOut='/home/jekim/workspace/jinju_ex/data/white1/result_all/video.mp4'

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
