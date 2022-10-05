import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() 

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of green color in HSV
    lower_green = np.array([25,75,75])
    upper_green = np.array([70,255,255])
    # Threshold the HSV image to get only green colors
    mask = cv.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break


    ret,thresh = cv.threshold(mask,127,255,0)

    contours,hierarchy = cv.findContours(thresh,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours= sorted(contours, key=cv.contourArea, reverse= True)
        #http://www.learningaboutelectronics.com/Articles/How-to-find-the-largest-or-smallest-object-in-an-image-Python-OpenCV.php
        # print("-")
        cnt = contours[0]
        M = cv.moments(cnt)

        epsilon = 0.1*cv.arcLength(cnt,True)
        x,y,w,h = cv.boundingRect(cnt)
        rectangle = cv.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)

    # Display the resulting frame
    cv.imshow('gray',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()