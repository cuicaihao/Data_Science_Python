# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:17:47 2018

@author: Caihao.Cui
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('simple.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()

cv2.imwrite('ShiTomas_Simple.png',img)