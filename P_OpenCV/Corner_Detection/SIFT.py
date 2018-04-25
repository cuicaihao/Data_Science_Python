# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:49:51 2018

@author: Caihao.Cui
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('Butterfly.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT()
# https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
sift = cv2.xfeatures2d.SIFT_create()

(kp, descs) = sift.detectAndCompute(gray, None)

img2 = cv2.drawKeypoints(gray, kp, None,(0,255,0),4)

plt.imshow(img2);
plt.show()


cv2.imwrite('sift_keypoints.jpg',img2)

#
img2=cv2.drawKeypoints(gray,kp,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints2.jpg',img2)

plt.imshow(img2);
plt.show()
#
#
#sift = cv2.SIFT()
#kp, des = sift.detectAndCompute(gray,None)