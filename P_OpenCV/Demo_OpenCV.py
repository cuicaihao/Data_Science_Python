import cv2
import numpy as np

print("This is a demo of openCV version:")
print(cv2.__version__)

img = cv2.imread('lena_std.tif')
cv2.imshow("lena",img)
cv2.waitKey(1000)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("lena_gray", gray)

k = cv2.waitKey(0)
if k==27:
	cv2.destroyAllWindows()
elif k == ord('s'): 	
	cv2.imwrite('leno_gray.tif', gray)
	cv2.destroyAllWindows()

	



