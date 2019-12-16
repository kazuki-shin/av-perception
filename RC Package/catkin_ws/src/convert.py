import subprocess, yaml
import rosbag
from cv_bridge import CvBridge
import cv2, os, pdb
import numpy as np

# make sure to create the output folders before calling
def rosbag_to_frames(testInputBagFile,read_topics,outputPath):
	# FX = 1.0
	# FY = 1.0

	info_dict = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', testInputBagFile], stdout=subprocess.PIPE).communicate()[0])

	bag = rosbag.Bag(testInputBagFile)
	print(bag)

	for topic_idx in range(len(read_topics)):
		genBag = bag.read_messages(read_topics[topic_idx])
		print('../data/output'+outputPath[topic_idx])
		for k,b in enumerate(genBag):
			print "OK, %d / %d" % (k, info_dict['messages'])
			cb = CvBridge()
			cv_image = cb.imgmsg_to_cv2( b.message, b.message.encoding )
			cv2.imwrite('../data/output'+outputPath[topic_idx] + '/' + str(k) + '.jpg', cv_image)
	bag.close()

def frames_to_mp4(input_folders):
	for topic_idx in range(len(input_folders)):
		img_array = []
		input_path = '../data/output' + input_folders[topic_idx] +'/'
		for count in range(len(os.listdir(input_path))):
			#pdb.set_trace()
			filename = input_path + str(count) + '.jpg'
			img = cv2.imread(filename)
			height, width, layers = img.shape
			size = (width,height)
			img_array.append(img)

			#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
			fourcc = 0x00000021
			fps = 30
			#pdb.set_trace()
			# '../data/output'+input_folders[topic_idx]+'.mp4'
			out = cv2.VideoWriter('../data/output'+input_folders[topic_idx]+'.mp4',fourcc,fps, size)

		for i in range(len(img_array)):
			out.write(img_array[i])
		out.release()
		print(input_folders[topic_idx][1:]+'vid done')

if __name__ == "__main__":
	testInputBagFile = '../data/cam_data_1.bag'
	read_topics = ['/cam_1','/cam_2','/cam_3']
	outputPath = ['/bag1_cam1','/bag1_cam2','/bag1_cam3']
	rosbag_to_frames(testInputBagFile,read_topics,outputPath)
	print('frames done')
	input_folders = ['/bag1_cam1','/bag1_cam2','/bag1_cam3']
	frames_to_mp4(input_folders)
