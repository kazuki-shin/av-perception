import cv2
import numpy as np
import matplotlib.pylab as plt
import pdb
import math
# pdb.set_trace()

def lengthh(line):
	x1,y1,x2,y2 = line[0], line[1], line[2], line[3]
	return math.sqrt((x2-x1)**2+(y2-y1)**2)

def isNotHor(line):
	x1,y1,x2,y2 = line[0], line[1], line[2], line[3]
	delx = abs(x1 - x2)
	dely = abs(y1 - y2)
	ang_rad = math.atan2(dely, delx)
	deg = math.degrees(ang_rad)
	if deg < 10 or (deg > 75):
		return False
	return True

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

def line_intersection(line1, line2):
	if line1 is None or line2 is None:
		return None, None
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
	   return None, None

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y

# read
# cap = cv2.VideoCapture(0)

fourcc = 0x00000021
fps = 30
# out = cv2.VideoWriter("outvid2.mp4", fourcc, fps, (640,480))

for i in range(972): # change loop to match video frames
	print(i)
	img = cv2.imread("bag2_cam2/" + str(0) + ".jpg")
	# ret, img = cap.read()
	#shape
	height = img.shape[0]
	width = img.shape[1]

	# area within image that will contain road, should change to match input vid
	vertices = np.array([[
				(0,362),
				(317,186),
				(424,186),
				(width,325),
				(width,height),
				(0, height)
			]], dtype=np.int32)


	# create mask using vertices
	mask = np.zeros_like(img)
	match_mask_color = 255
	cv2.fillPoly(mask, vertices, match_mask_color)
	cv2.imshow("mask",mask)
	# create canny image, can change threshold vals
	canny_image = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 25, 50)
	cv2.imshow("can",canny_image)
	# masked image that lines can be seen in
	masked_image = cv2.bitwise_and(canny_image, mask[:,:,0])
	cv2.imshow("masked",masked_image)
	# detect lines
	lsd = cv2.createLineSegmentDetector(0)
	lines = lsd.detect(masked_image)[0] #Position 0 of the returned tuple are the detected lines

	# change line_min in correspondence w img size
	line_min = 30
	ct = 0
	# longest = None
	# l_len = -1 * float("inf")
	# sec_longest = None
	# sl_len = -1 * float("inf")

	for line in lines:
		if lengthh(line[0]) > line_min and isNotHor(line[0]):
			ct += 1
	# pdb.set_trace()

	new_lines = np.zeros((ct, lines.shape[1],lines.shape[2]), dtype=np.int32)

	j = 0
	for i in range(lines.shape[0]):
		line = lines[i]
		if lengthh(line[0]) > line_min and isNotHor(line[0]):
			new_lines[j] = line
			j += 1

		# if lengthh(line[0]) > l_len:
		# 	sl_len = l_len
		# 	l_len = lengthh(line[0])
		# 	sec_longest = longest
		# 	longest = line[0]
		# elif lengthh(line[0]) > sl_len:
		# 	sl_len = lengthh(line[0])
		# 	sec_longest = line[0]

	# A = (longest[0], longest[1])
	# B = (longest[2], longest[3])
	# C = (sec_longest[0], sec_longest[1])
	# D = (sec_longest[2], sec_longest[3])
	# intx, inty = line_intersection((A, B), (C, D))
	# print(intx, inty)
	# mid_x = width / 2
	# turn_dir = ""
	# if intx is None:
	# 	if intx > mid_x + 500 or intx < mid_x - 500:
	# 		turn_dir = ""
	# 	else:
	# 		if intx > mid_x:
	# 			turn_dir = "right"
	# 		else:
	# 			turn_dir = "left"
	# Draw detected lines in the image
	# drawn_img = lsd.drawSegments(img,new_lines)
	drawn_img = draw_the_lines(img, new_lines)

	# drawn_img = cv2.putText(drawn_img, turn_dir, (width // 4,height // 5), cv2.FONT_HERSHEY_SIMPLEX,  
 #                   1, (255, 0, 0), 2, cv2.LINE_AA)

	# cv2.imwrite("out/" + str(i) + ".jpg", drawn_img)
	# out.write(drawn_img)
	# Show image
	cv2.imshow("LSD",drawn_img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# cap.release()
# out.release()
cv2.destroyAllWindows()
