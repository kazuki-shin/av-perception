import numpy as np
import cv2

def redOnImage():
	mario = cv2.imread("/Users/aneeshmysore/Desktop/Fall2019/CS445/CS445_Final_Project/stop_sign.jpg")
	hsv = cv2.cvtColor(mario, cv2.COLOR_BGR2HSV) 
	lower_red = np.array([160,100,100]) 
	upper_red = np.array([179,255,255])

	mask = cv2.inRange(hsv, lower_red, upper_red)

	res = cv2.bitwise_and(mario,mario, mask= mask) 
	cv2.imshow('mario',mario)
	
	cv2.imshow('mask',mask) 
	cv2.imshow('res',res)
	cv2.waitKey(20*1000)
	cv2.destroyAllWindows()

redOnImage()
