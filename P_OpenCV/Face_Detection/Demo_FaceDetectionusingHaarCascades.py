import cv2
# Face Detection using Haar Cascades
# code source: https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html

print(cv2.__version__) 


face_cascade_path =  r"./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
 
eye_cascade_path =  r"./opencv-master/data/haarcascades/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)


img = cv2.imread('MyLTUTeam.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('MyLTUTeam_gray.tif',gray)
    cv2.destroyAllWindows()
 
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
       #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('MyLTUTeam_fd.tif',img)
    cv2.destroyAllWindows()
    
cv2.waitKey(10000)
cv2.destroyAllWindows()
