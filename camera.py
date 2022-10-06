# Code references:
# http://www.learningaboutelectronics.com/Articles/How-to-find-the-largest-or-smallest-object-in-an-image-Python-OpenCV.php

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
    lower_green_hsv = np.array([25,75,75])
    upper_green_hsv = np.array([70,255,255])
    # define range of green color in BGR
    lower_green_bgr = np.array([50,180,60])
    upper_green_bgr = np.array([170,255,180])
    # define range of green color in HSV for Color Picker. Test Color = HSV(65,153,200)
    lower_green_cp = np.array([40,100,90])
    upper_green_cp = np.array([90,255,255])

    # Threshold the HSV image to get only green colors
    mask_hsv = cv.inRange(hsv, lower_green_hsv, upper_green_hsv)
    # Threshold the BGR image to get only green colors
    mask_bgr = cv.inRange(frame, lower_green_bgr, upper_green_bgr)
    # Threshold the HSV image to get only green colors for Color Picker
    mask_cp = cv.inRange(hsv, lower_green_cp, upper_green_cp)
    
    # Bitwise-AND mask and original image HSV
    res_hsv = cv.bitwise_and(frame,frame, mask= mask_hsv)
    res_bgr = cv.bitwise_and(frame,frame, mask= mask_bgr)
    res_cp = cv.bitwise_and(frame,frame, mask= mask_cp)

    gray_hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_bgr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_cp = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break


    ret_hsv,thresh_hsv = cv.threshold(mask_hsv,127,255,0)
    ret_bgr,thresh_bgr = cv.threshold(mask_bgr,127,255,0)
    ret_cp,thresh_cp = cv.threshold(mask_cp,127,255,0)

    contours_hsv,hierarchy_hsv = cv.findContours(thresh_hsv,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_bgr,hierarchy_bgr = cv.findContours(thresh_bgr,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_cp,hierarchy_cp = cv.findContours(thresh_cp,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours_hsv) > 0:
        contours_hsv = sorted(contours_hsv, key=cv.contourArea, reverse= True)
        #http://www.learningaboutelectronics.com/Articles/How-to-find-the-largest-or-smallest-object-in-an-image-Python-OpenCV.php
        cnt_hsv = contours_hsv[0]
        epsilon_hsv = 0.1*cv.arcLength(cnt_hsv,True)
        x,y,w,h = cv.boundingRect(cnt_hsv)
        rectangle = cv.rectangle(gray_hsv,(x,y),(x+w,y+h),(0,255,0),2)

    if len(contours_bgr) > 0:
        contours_bgr = sorted(contours_bgr, key=cv.contourArea, reverse= True)
        #http://www.learningaboutelectronics.com/Articles/How-to-find-the-largest-or-smallest-object-in-an-image-Python-OpenCV.php
        cnt_bgr = contours_bgr[0]
        epsilon_bgr = 0.1*cv.arcLength(cnt_bgr,True)
        x,y,w,h = cv.boundingRect(cnt_bgr)
        rectangle = cv.rectangle(gray_bgr,(x,y),(x+w,y+h),(0,255,0),2)

    if len(contours_cp) > 0:
        contours_bgr = sorted(contours_cp, key=cv.contourArea, reverse= True)
        #http://www.learningaboutelectronics.com/Articles/How-to-find-the-largest-or-smallest-object-in-an-image-Python-OpenCV.php
        cnt_cp = contours_cp[0]
        epsilon_cp = 0.1*cv.arcLength(cnt_cp,True)
        x,y,w,h = cv.boundingRect(cnt_cp)
        rectangle = cv.rectangle(gray_cp,(x,y),(x+w,y+h),(0,255,0),2)

    # Display the resulting frame
    cv.imshow('HSV Green Object Tracking',gray_hsv)
    cv.imshow('BGR Green Object Tracking',gray_bgr)
    cv.imshow('Narrow Threshold Range for Color Picker Tracking',gray_cp)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
