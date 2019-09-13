# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 13:50:30 2018

@author: manishre
"""
#%% Libraries
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
import copy
#%% Functions

def padding(image):
    width = image.shape[1]
    height = image.shape[0]
    pad_image = np.asarray([[255 for x in range (width+2)] for y in range (height+2)])
    pad_image[1:-1,1:-1] = image
#    cv2.imshow('padded',np.asarray(pad_image,dtype = 'uint8'))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return np.asarray(pad_image,dtype = 'uint8')

def dilation(image,SE):
    D = image.copy()
    width = image.shape[1]
    height = image.shape[0]
    for i in range(0,height):
        for j in range(0,width):
            ao =[]
            for kr in range(0,3):
                for kc in range(0,3):
                     #ao.append(SE[kr,kc]*image[i-kr,j-kc])
                    a = image[i-kr,j-kc]
                    S_E = SE[kr,kc]
                    ao.append(a & S_E)      
            D[i,j]=max(ao)
#    cv2.imshow('Dilation image',np.asarray(D,dtype ='uint8'))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()          
    return np.asarray(D,dtype='uint8')
def erosion(image,SE):
    E = image.copy()
    width = image.shape[1]
    height = image.shape[0]
    for i in range(0,height):
        for j in range(0,width):
            ao =[]
            for kr in range(0,3):
                for kc in range(0,3):
                    a = image[i-kr,j-kc]
                    S_E = SE[kr,kc]
                    if (S_E==255 and a==S_E):
                        ao.append(255)
                    else:
                        ao.append(0)
            E[i,j]=min(ao)             
#    cv2.imshow('erosion image',np.asarray(E,dtype ='uint8'))           
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return np.asarray(E,dtype ='uint8')  

def similarity(im_o, im_c):
	mse = np.sum((im_o.astype("float") - im_c.astype("float")) ** 2)
	mse /= float(im_o.shape[0] * im_o.shape[1])
	return mse

def boundary(img_e,kernel):
    img_ero = erosion(img_e,kernel)
    img_dil = dilation(img_e,kernel)
    img_bound = np.subtract(img_dil,img_ero)
    return img_bound
#def opening():

#%% MAIN LOOP

#%% Loading the Images
image = cv2.imread('noise.jpg',0)
img = image.copy()
padded_image = padding(img)
cv2.imshow('orginal image',image)
cv2.waitKey(0)
cv2.imshow('Padded image',padded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('dimensions',padded_image.shape)
# structring Element
kernel = np.matrix([[255,255,255],[255,255,255],[255,255,255]])
#kernel = np.matrix([[255,255,255,255,255],[255,255,255,255,255],[255,255,255,255,255],[255,255,255,255,255],[255,255,255,255,255]])
#%% PART 1

# Closingg
img0 = dilation(img,kernel)
img2= erosion(img0,kernel)
img3 = erosion(img2,kernel)
cv2.imwrite('res_noise1.jpg',np.asarray(img3))
# Opening
img4 = erosion(img,kernel)
img5 = dilation(img4,kernel)
img6 = dilation(img5,kernel)
cv2.imwrite('res_noise2.jpg',img6)
# displaying the images
cv2.imshow('closing',img3)
cv2.waitKey(0)
cv2.imshow('opening',img6)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% PART 2

# Finding comparisons between two images  by using mean square error
# Condtion1: if the both images are same then the error will be 0 
# Conditon2: if the both images are not same then the value of error will be increasing
#-----------------------------
# For the variable o below we are caluclating the comparing the logic by passing same image
o = similarity(img,img)
# Caluclaing the similarity between two resultant images
s = similarity(img3,img6)
print('s',s)
if o == 0:
    print('The two images are same')
else :
    print('The two images are not same')
if s == 0:
    print('The two images are same')
else :
    print('The two images are not same')


#%% PART 3
## The difference between dilation and erosion of a image results in the boundary of a image
img_c = img3.copy()    
img_o = img6.copy()
img_boundo = boundary(img_o,kernel)
img_boundc = boundary(img_c,kernel)
cv2.imshow('boundary open',np.asarray(img_boundo,dtype='uint8'))
cv2.waitKey(0)
cv2.imshow('boundary close',np.asarray(img_boundc,dtype='uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('res_bound1.jpg',img_boundo)
cv2.imwrite('res_bound2.jpg',img_boundc)
#img_bound = np.asarray(img_e)/
#img_bound = img_bound/np.max(img_bound)
#%%
s1 = similarity(img_boundc,img_boundo)
print(s1)