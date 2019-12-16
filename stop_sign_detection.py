import numpy as np
import cv2
import os

def detectingContour():
	img = cv2.imread("/Users/aneeshmysore/Desktop/Fall2019/CS445/CS445_Final_Project/stop_sign.jpg")

	ret, thresh1 = cv2.threshold(img[:,:,0], 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	_,contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	for contour in contours:
	    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
	    if len(approx) == 8:
	        cv2.drawContours(img, [contour], 0, (0, 255, 0), 6)

	cv2.imshow('sign', img)       
	cv2.waitKey(0)
	cv2.destroyAllWindows()

detectingContour()
