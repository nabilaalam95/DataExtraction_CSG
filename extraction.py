import cv2
import numpy as np
import matplotlib.pyplot as plt

im_annot=np.load('bird_view_130_annotation.npy')
im_color=cv2.imread("bird_view_130_color.jpeg")
lap = cv2.Laplacian(im_annot,cv2.CV_64F,ksize=3)
lap = np.uint8(np.absolute(lap))
for filt in range(lap.shape[2]):
    for x in range(lap.shape[0]):
        for y in range(lap.shape[1]):
            if lap[x,y,filt]!=0:
                lap[x,y,0]=255
                lap[x,y,1]=255
                lap[x,y,2]=255
plt.figure(figsize=(20,15))
plt.imshow(lap)
for filt in range(lap.shape[2]):
    for x in range(lap.shape[0]):
        for y in range(lap.shape[1]):
            if lap[x,y,filt]!=0:
                im_color[x,y,0]=255
                im_color[x,y,1]=0
                im_color[x,y,2]=0
plt.figure(figsize=(20,15))
plt.imshow(im_color[:,:,::-1])
for filt in range(im_annot.shape[2]):
    for x in range(im_annot.shape[0]):
        for y in range(im_annot.shape[1]):
            if im_annot[x,y,filt]!=0:
                im_annot[x,y,0]=1
                im_annot[x,y,1]=1
                im_annot[x,y,2]=1
im_color_arr=np.array(im_color)
out=np.multiply(im_color,im_annot)
plt.figure(figsize=(20,15))
plt.imshow(out[:,:,::-1])
