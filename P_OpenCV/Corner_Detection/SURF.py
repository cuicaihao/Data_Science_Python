# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:52:26 2018

@author: Caihao.Cui
"""

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
 
surf = cv2.xfeatures2d.SURF_create(400)
#surf = cv2.SURF(400)
(kp, des) = surf.detectAndCompute(gray,None)

print(len(kp))

img2 = cv2.drawKeypoints(img,kp[1:150],None,(255,0,0),4)
plt.imshow(img2)
plt.title('150 points')
plt.show()


cv2.imwrite('SURF_keypoints.jpg',img2)


img2=cv2.drawKeypoints(img,kp,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('SURF_keypoints2.jpg',img2)

plt.imshow(img2);
plt.show()
#