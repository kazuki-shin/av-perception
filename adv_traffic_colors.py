import os
import cv2
import numpy as np
import sys

def generalColorDetection(circle_arr, num_rows, num_cols,  mask):
    #No circles detected
    if circle_arr is None:
        return None
    else:
        ret = []
        #convert to integers for x,y coordinates
        circle_arr = circle_arr.astype(int)

        #finds the amount of red, green, and yellow pixels in the circles it found
        for xyr in circle_arr[0, :]:
            if xyr[0] <= num_cols and xyr[1] <= num_rows: 
                correct_col = 0.0
                point = 0.0
                #iterate around the center point to ensure there are pixels of the same color
                for y in range(-4, 4, 1):
                    for x in range(-4, 4, 1):
                        if xyr[0] + x < num_cols and xyr[1] + y < num_rows:
                            correct_col +=  mask[xyr[1]+ y, xyr[0]+ x]

                if correct_col / 64 > 50:
                    new_arr = [xyr[0], xyr[1], xyr[2]]
                    ret.append(new_arr)
    return np.asarray(ret)

def createMask(hsv_scaled):
    #Create the red, blue, and green masks
    lower_red = np.array([0,100,100])
    upper_red = np.array([10,255,255])

    #Upper bound red
    up_lower_red = np.array([10,50,50])
    up_higher_red = np.array([180,255,255])

    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])

    lower_yellow = np.array([15, 150, 150])
    upper_yellow = np.array([32,255,255])

    #Create the masks
    red1_mask = cv2.inRange(hsv_scaled, lower_red, upper_red)
    red2_mask = cv2.inRange(hsv_scaled, up_lower_red, up_higher_red)
    red_mask = red1_mask + red2_mask
    
    green_mask = cv2.inRange(hsv_scaled, lower_green, upper_green)
    
    yellow_mask = cv2.inRange(hsv_scaled, lower_yellow, upper_yellow)

    return red_mask, green_mask, yellow_mask


def detectCircles(red_mask, green_mask, yellow_mask):

    #Red, blue and yellow circle detectors
    red = cv2.HoughCircles(red_mask, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=10, minRadius=0, maxRadius=30)

    green = cv2.HoughCircles(green_mask, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10, minRadius=0, maxRadius=30)

    yellow = cv2.HoughCircles(yellow_mask, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=5, minRadius=0, maxRadius=30)

    return red, green, yellow

def drawCircles(red, yellow, green, output, font):
    if red is not None and len(red) != 0:
        cv2.circle(output, (red[0][0], red[0][1]), red[0][2]+5, (0, 255, 0), 2)
        cv2.putText(output,'RED',(red[0][0], red[0][1]), font, 10,(255,0,0),4)

    if yellow is not None and len(yellow) != 0:
        cv2.circle(output, (yellow[0][0], yellow[0][1]), yellow[0][2]+5, (0, 255, 0), 2)
        cv2.putText(output,'YELLOW',(yellow[0][0], yellow[0][1]), font, 10,(255,0,0),4)

    if green is not None and len(green) != 0:
        cv2.circle(output, (green[0][0], green[0][1]), green[0][2]+5, (0, 255, 0), 2)
        cv2.putText(output,'GREEN',(green[0][0], green[0][1]), font, 10, (255,0,0), 4)

    return

def setUpImage():

    image = cv2.imread("/Users/aneeshmysore/Desktop/Fall2019/CS445/CS445_Final_Project/green_traffic_light.jpeg")
    output = image.copy()

    hsv_scale_convert = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v1 = cv2.split(hsv_scale_convert)

    r_mask,g_mask,y_mask = createMask(hsv_scale_convert)
    red_circles, green_circles, yellow_circles =  detectCircles(r_mask, g_mask, y_mask)

    num_rows, num_cols = image[:,:, 0].shape
    font = cv2.FONT_HERSHEY_PLAIN
    red_in_im = generalColorDetection(red_circles, num_rows, num_cols, r_mask)
    yellow_in_im = generalColorDetection(yellow_circles, num_rows, num_cols, y_mask)
    green_in_im= generalColorDetection(green_circles, num_rows, num_cols, g_mask)

    drawCircles(red_in_im, yellow_in_im, green_in_im, output, font)

    cv2.imshow('detected results', output)
    cv2.imwrite('finalCircleDetection.png', output)
    cv2.waitKey(10 * 1000)
    cv2.destroyAllWindows()

#Starts the calls
setUpImage()
