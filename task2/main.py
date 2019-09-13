# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:14:24 2018

@author: manishre
"""

import cv2
import numpy as np
import scipy.misc

#%% FUNCTIONS
def con(img,kernel):
    img1 =img.copy()
    height,width = img.shape
    for x in range (0,height):
        for y in range (0,width):
            su_x=0
            for k in range (0,3):
                for l in range (0,3):
                    a= img[x-k,y-l]
                    w1 = kernel[k][l]
                    su_x = su_x + (w1*a);
            img1[x,y] = su_x
            
    print(img1.shape)
    return img1

def Thresholding(img,tv,threshold):
    img_T = img.copy()
    height,width = img.shape
    if(threshold=='b'):
        for i in range(0,height):
            for j in range(0,width):
                if (img_T[i,j]>tv):
                    img_T[i,j] =255 
                else:
                    img_T[i,j] =0
    else:
        for i in range(0,height):
            for j in range(0,width):
                if (img_T[i,j]>tv):
                    img_T[i,j] =0 
                else:
                   img_T[i,j] =255
    return img_T

#%% IMREAD
image = scipy.misc.imread('point1.jpg',1)
#image = cv2.imread('point1.jpg',0)
img1 = image.copy()
kernel = [[-1,-1,-1],
          [-1,8,-1],
          [-1,-1,-1]]


#%%
image_con = con(img1,kernel)
#%%
image_con=  np.abs(image_con)/np.max(np.abs(image_con))
#%%image_con
image_thre = Thresholding(image_con,0.5,'b')
#cv2.circle(image_thre, center = (point[0],point[1]), radius = 12, color=(255,255,255), thickness = 1)
#%% Point Detection 
point_coordinates =np.unravel_index(image_thre.argmax(),image_thre.shape)
print('The point is detected at:',point_coordinates)
cv2.circle(image_thre,(point_coordinates[1],point_coordinates[0]), radius = 12, color=(255,255,255), thickness = 1)
cv2.putText(image_thre, str(point_coordinates),(point_coordinates[1]+6,point_coordinates[0]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
#%%
#scipy.misc.imsave('img_con.jpg',image_con)
cv2.imwrite('task2a_output.jpg',image_thre)
cv2.imshow('img_o',image_con)
cv2.waitKey(0)
cv2.imshow('img_t',image_thre)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% TASK2B
image_2B = cv2.imread('segment.jpg',0)
#%%

intensity = np.zeros(256)
h,w = image_2B.shape

for i in range(0,h):
    for j in range(0,w):
        intensity[image_2B[i,j]] += 1

#%%
    intensity = intensity.astype(int)
    print(intensity)
#%%
    intensity[0]=0
    intensity[0]=max(intensity)+100
    
#%%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(intensity)
plt.xlabel('pixel value')
plt.ylabel('intensity')
plt.savefig('task2a_histogram.jpg')
plt.show()
#%%
np.set_printoptions(threshold=np.nan)
image_output=image_2B.copy()
#print(image_2B)
print('====================')

image_i = Thresholding(image_2B,205,'b')
image_i1 = Thresholding(image_2B,203,'w')
#%%
#%%
row_max= 321
row_min = 0
col_max = 712
col_min =0
for i in range(0,h):
    for j in range(0,w):
       if image_i[i][j]>205:
          if row_max>i:
              row_max =i
          elif col_max > j:
              col_max =j
          elif row_min < i:
              row_min = i
          elif col_min < j:
              col_min =j
#%%
cv2.rectangle(image_i,(col_max,row_max),(col_min,row_min),(255,255,255),2)
cv2.putText(image_i, str((row_min,col_min)),(col_min+6,row_min+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(image_i, str((row_max,col_max)),(col_max-80,row_max+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(image_i, str((row_min,col_max)),(col_max-80,row_min+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(image_i, str((row_max,col_min)),(col_min+6,row_max+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
      
 #%%
# show the output image
cv2.imwrite('task2b_output.jpg',image_i)
cv2.imshow('image_2B',image_2B)
cv2.waitKey(0)
cv2.imshow('image_output',image_i)
cv2.waitKey(0)
cv2.destroyAllWindows()
