# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:43:37 2018

@author: manishre
"""

#%% LIBRARIES
import cv2
import numpy as np
import copy
import math

#%%Reading the image
image = cv2.imread('A:\\classes\\Intro to Computer vision and Image Processing\\projrct\\project3\\task3\\hough.jpg',0)
width = image.shape[1]
height = image.shape[0]
print("Width :",width)
print('Height:',height)
#%% Functions
def edge_detection(pad_image):
    s_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
    s_y =[[1,2,1],[0,0,0],[-1,-2,-1]]
    #Flip Operations
    for i in range (0,3):
        temp =s_x[i][0]
        s_x[i][0] =s_x[i][2]
        s_x[i][2] =temp
    print('s_x',s_x)
    for y in range (0,3):
        temp = s_y[0][y]
        s_y[0][y] =s_y[2][y]
        s_y[2][y] =temp
    img_x = np.asarray([[0 for x in range(0,width)] for y in range(0,height)])
    img_y = np.asarray([[0 for x in range(0,width)] for y in range(0,height)])
    # Convolution 
    for x in range (0,height):
        for y in range (0,width):
            su_x=0
            su_y =0
            for k in range (0,3):
                for l in range (0,3):
                    a= pad_image[x-k,y-l]
                    w1 = s_x[k][l]
                    w2 = s_y[k][l]
                    su_x = su_x + (w1*a);
                    su_y = su_y + (w2*a)
            img_x[x,y] = abs(su_x)
            img_y[x,y] = abs(su_y)
    print(img_x)
    d = img_x.shape
    print('edge_'+str(i)+'width, height',d)
    return np.array(img_x,dtype='uint8'),np.array(img_y,dtype='uint8')




def hough_transform(img_bin):
  nR,nC = img_bin.shape
  theta = np.linspace(-90.0, 0.0, np.ceil(90.0) + 1.0)
  theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
  print('1start')
  D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
  q = np.ceil(D)
  nrho = 2*q + 1
  rho = np.linspace(-q, q, nrho)
  H = np.zeros((len(rho), len(theta)))
  for rowIdx in range(nR):
    for colIdx in range(nC):
      if img_bin[rowIdx, colIdx]:
        for thIdx in range(len(theta)):
          rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + \
              rowIdx*np.sin(theta[thIdx]*np.pi/180)
          rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
          H[rhoIdx[0], thIdx] += 1
  print('1end')        
  return rho, theta, H


#%%

pad_image = np.asarray([[0 for x in range (width+2)] for y in range (height+2)])
pad_image[1:-1,1:-1] = image
cv2.imshow('padded',np.asarray(pad_image,dtype = 'uint8'))
cv2.waitKey(0)

#g_x = np.asarray([[0 for x in range(0,width)] for y in range(0,height)])
g_x,g1 = edge_detection(pad_image)
cv2.imshow('g_x',abs(g_x))
cv2.waitKey(0)
cv2.imshow('g_y',abs(g1))
cv2.waitKey(0)

cv2.destroyAllWindows()
#g_y = np.asarray([[0 for x in range(0,width)] for y in range(0,height)])


#%%
im =image.copy()
edges = cv2.Canny(im, threshold1 = 0, threshold2 = 50, apertureSize = 3)
r,t,h = hough_transform(edges)

#%%
P =r
angle = t
space =h
#%%

def sort(P_space, n, rhos, thetas):

  print('2start')
  unstack = list(set(np.hstack(P_space)))
  unsorte = sorted(unstack, key = lambda n: -n)
  points = [(np.argwhere(P_space == x)) for x in unsorte[0:n]]
  rho_theta = []
  x_y = []
  for i in range(0, len(points), 1):
    co = points[i]
    for i in range(0, len(co), 1):
      n,m = co[i] # n by m matrix
      rho = rhos[n]
      theta = thetas[m]
      rho_theta.append([rho, theta])
      x_y.append([m, n]) # just to unnest and reorder coords_sorted
  print('2end')  
  return [rho_theta[0:n], x_y]
#%%
  val,val1 = sort(space, 70, P, t)
  #%%
  print(val)
  #%%
def bound(pt, ymax, xmax):
  x, y = pt

  if x <= xmax and x >= 0 and y <= ymax and y >= 0:
    return True
  else:
    return False

def roundnum(tup):
  x,y = [int(round(num)) for num in tup]
  return (x,y)

def draw_Lines(target_im, pairs):
    tar0 = target_im.copy()
    tar1 = target_im.copy()
    h,w = np.shape(target_im)
    th0=[]
    print('len',len(pairs))
    pts0X=[]
    pts1X=[]
    pts0Y=[]
    pts1Y=[]
    for i in range(1,len(pairs),1):
        point =pairs[i]
        rho = point[0]
        theta = point[1]
        theta = point[1] *np.pi/180
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        left = (0, b)
        right = (w, w * m + b)
        top = (-b / m, 0)
        bottom = ((h - b) / m, h)
        #print(m)
        if (m>3):
           # print('greater')
           print('top',top)
           print('left',left)
           print('right',right)
           print('bottom',bottom)
           pts =[pt for pt in [left, right, top, bottom] if bound(pt, h, w)]
           pts0X.append(roundnum(pts[0]))
           pts1X.append(roundnum(pts[1]))
        else:
            #print('less')
            pts1 =[pt for pt in [left, right, top, bottom] if bound(pt, h, w)]
            pts0Y.append(roundnum(pts1[0]))
            pts1Y.append(roundnum(pts1[1]))
       
    return pts0X,pts1X,pts0Y,pts1Y   
    
 #%%
p0,p1,p2,p3= draw_Lines(image,val)
#%%
imgr =image.copy()
imgb =image.copy()
for i in range(0,len(p0)):
 cv2.line(imgr, p0[i],p1[i] , (255,255,255), 2)
 cv2.line(imgb,p2[i],p3[i],(255,0,0),2)
#%%
cv2.imshow('red lines',imgr)
cv2.waitKey(0)
  
cv2.imshow('blue lines',imgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
  #%%
hh,ww = image.shape
for v in range(0,len(val1)):
     temp = val1[v]
     uuu= []
     if(temp[0]<hh and temp[1]<ww):
         uuu.append(temp)
  
#%%
print(p0)
print(p1)  #%%
