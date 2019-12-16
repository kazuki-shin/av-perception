import numpy as np
import cv2
import os

def trafficLightDetectorBasic():
	image = cv2.imread("/Users/aneeshmysore/Desktop/Fall2019/CS445/CS445_Final_Project/traffic-light.jpg")
	output = image.copy()

	#HoughCircles requires a gray scale image 
	gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray_scale, cv2.HOUGH_GRADIENT, 1.2, 100)
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = circles.astype(int)
		for (x, y, r) in circles[0,:]:
			# draw the circle in output image
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
	 
	# show output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(20*1000)
	cv2.destroyAllWindows()

trafficLightDetectorBasic()