import cv2
import numpy as np
import matplotlib.pylab as plt
import pdb
import math
# pdb.set_trace()

def lengthh(x1,y1,x2,y2):
	return math.sqrt((x2-x1)**2+(y2-y1)**2)

def isNotHor(x1,y1,x2,y2):
	delx = abs(x1 - x2)
	dely = abs(y1 - y2)
	ang_rad = math.atan2(dely, delx)
	deg = math.degrees(ang_rad)
	if deg < 10:
		return False
	return True

# read
img = cv2.imread("floor.jpg")

#s hape
height = img.shape[0]
width = img.shape[1]

# area within image that will contain road, should change to match input vid
vertices = np.array([[
			(0, height),
			(0, height - 50),
			(width/2, 249),
			(width, height - 108),
			(width, height)
		]], dtype=np.int32)

# create mask using vertices
mask = np.zeros_like(img)
match_mask_color = 255
cv2.fillPoly(mask, vertices, match_mask_color)
# create canny image, can change threshold vals
canny_image = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 75, 150)

# masked image that lines can be seen in
masked_image = cv2.bitwise_and(canny_image, mask[:,:,0])

# detect lines
lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(masked_image)[0] #Position 0 of the returned tuple are the detected lines

# change line_min in correspondence w img size
line_min = 30
ct = 0
for line in lines:
	if lengthh(line[0][0],line[0][1],line[0][2],line[0][3]) > line_min and isNotHor(line[0][0],line[0][1],line[0][2],line[0][3]):
		ct += 1

new_lines = np.zeros((ct, lines.shape[1],lines.shape[2]), dtype=np.int32)

j = 0
for i in range(lines.shape[0]):
	line = lines[i]
	if lengthh(line[0][0],line[0][1],line[0][2],line[0][3]) > line_min and isNotHor(line[0][0],line[0][1],line[0][2],line[0][3]):
		new_lines[j] = line
		j += 1

print(lines.shape, new_lines.shape)

def draw_the_lines(img, lines):
	if lines is None:
		return img
	img = np.copy(img)
	blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(blank_image, (x1,y1), (x2,y2), (0, 0, 255), thickness=5)
	img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
	return img

# Draw detected lines in the image
# drawn_img = lsd.drawSegments(img,new_lines)
drawn_img = draw_the_lines(img, new_lines)

# Show image
cv2.imshow("LSD",drawn_img )
cv2.waitKey(15000)
