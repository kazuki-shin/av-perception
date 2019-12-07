import cv2
import numpy as np
import matplotlib.pylab as plt
#Read gray image
# img = cv2.imread("road2.png",0)
#
# #Create default parametrization LSD
# lsd = cv2.createLineSegmentDetector(0)
#
# #Detect lines in the image
# lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
#
# #Draw detected lines in the image
# drawn_img = lsd.drawSegments(img,lines)
#
# #Show image
# cv2.imshow("LSD",drawn_img )
# cv2.waitKey(0)
def region_of_interest(img, vertices):
	mask = np.zeros_like(img)
	#channel_count = img.shape[2]
	match_mask_color = 255
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_the_lines(img, lines):
	if lines is None:
		return img
	img = np.copy(img)
	blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(blank_image, (x1,y1), (x2,y2), (0, 0, 255), thickness=2)

	img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
	return img

image = cv2.imread('road1.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(image.shape)


def linesInVideo():
	cap = cv2.VideoCapture(0)
	while(True):
		# print("frame: ", i)
		ret, image = cap.read()
		# image = cv2.imread('frames2/' + 'thumb' + "{:04d}".format(i) + '.jpg')
		height = image.shape[0]
		width = image.shape[1]
		region_of_interest_vertices = [
			(0, height),
			(width/2, height/2),
			(width, height)
		]
		gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		canny_image = cv2.Canny(gray_image, 100, 200)
		cropped_image = region_of_interest(canny_image,
						np.array([region_of_interest_vertices], np.int32),)
		lines = cv2.HoughLinesP(cropped_image,
								rho=6,
								theta=np.pi/180,
								threshold=160,
								lines=np.array([]),
								minLineLength=40,
								maxLineGap=25)
		image_with_lines = draw_the_lines(image, lines)
		cv2.imshow("img", image_with_lines)
		# plt.show()

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

linesInVideo()
