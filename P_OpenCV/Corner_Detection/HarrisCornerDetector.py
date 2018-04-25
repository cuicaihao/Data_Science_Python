import cv2
import numpy as np
import matplotlib.pyplot as plt

#filename = 'chessboard.jpg'
filename = 'pump.png' 
#filename = 'simple.png' 
 
img = cv2.imread(filename)

plt.figure()
plt.imshow(img)
plt.show()


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]


print(dst.max())
print(dst>0.01*dst.max())

plt.figure()
plt.imshow(dst>0.01*dst.max(), 'gray')
plt.show()

plt.figure()
plt.imshow(img)
plt.show()

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()