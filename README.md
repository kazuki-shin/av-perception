# 445 Final Project: Autonomous Driving 

# 1. Our Team
Kazuki Shin, Computer Engineering, May 2021\
Rishav Rout, Computer Science, May 2021\
Aneesh Mysore, Computer Science, December 2020\
Kshitij Gupta, Computer Science, May 2020

# 2. Project Topic
We want to implement the perception side of autonomous vehicles. The problem we want to solve is safety of driver assistance systems in modern level 2 autonomous vehicles. Our goal is to create an RC car that is able to reliably detect objects in a traffic based environments and react accordingly. This would include tasks such as image stitching, image enhancement, lane tracking, and human/traffic signal detection.

# 3. Project Motivation
There are many fatalities and casualties that occur daily from traffic accidents. Improving perception systems that can Our main motivation for this project is to improve ADAS through use of opencv and ROS. Creating features such as lane keeping and driver detection will help decrease the amount of accidents that occur throughout the world.

I (Kazuki) am interested in driver attention detection within the vehicle since I believe that this is a big factor in evaluating the safety of ADAS systems. Even if there is an advanced system that can navigate well through traffic, driver intervention will be necessary in crucial situations which the system cannot react to.

I (Kshitij) think that this will be very interesting to work on because I have always been fascinated by autonomous cars. There is a lot of exciting work being done in this area currently and I would love to explore this area.

I (Aneesh) think that this is a very interesting topic because I have never actually worked with autonomous cars. What fascinates me about autonomous cars is the technology behind them and how this technology can be so helpful in other areas. There are also many obvious financial benefits such as in regards to car insurance, etc. I want to work in this area during my career and so I would love to explore more into this topic. 

I (Rishav) think this would be really cool topic to work on since autonomous driving is something that people point to when they talk about computer science, especially within artificial intelligence. Being able to work on something like this, and understanding how it works internally, is part of the reason I wanted to major in computer science. 

# 4. Resources
Remote control car 
Jetson TX2 for image processing
Arduino for control command to motor output 
Computer with GPU 
Battery to power the RC and Jetson

# 5. Approach

Step 1: Data Input. Get data from all 3 logitech cameras by setting up ROS nodes, topics so we can publish and subscribe to the master, allowing us to get image data. (Kazuki)

Step 2: Video Stitching. We create a module that stitches images from three cameras that are fixed on the RC car. We will be using methods similar to the resources linked below to stitch the images from cameras attached at different angles to get wide angled view of the surroundings of the RC car. Because of this there will be distortion created in the image and we will be correcting this as described later. (Kshitij) 
https://github.com/ppwwyyxx/OpenPano 
https://github.com/ziqiguo/CS205-ImageStitching 

Step 3: Image processing and Enhancement. We create a module that takes in stitched images from the previous module and applies various image enhancement techniques like color and contrast enhancement. We are going to be using our implementation from project to perform this part. (Aneesh) 

Step 4: Lane Tracking. In order to do lane tracking we will first start by correcting the distortion that occurs when data from the 3-D world are transformed into 2-D images. The way we plan to do this is by using Python’s OpenCV library that address two common types of distortions: 1) radial distortion and 2) tangential distortion. (Aneesh)

Step 5: The next step will be to take the undistorted image and use the line detector in Python’s OpenCV. We will use the HoughLine transform method to do such a detection. An alternative approach to using the line detector would be to use color channels and apply threshold filters to highlight the lane points. This could be done by using HSV or HSL (alternate representations of the RGB model) to highlight lane points. We could improve upon this by using a machine learning model to improve understanding on what lines are actually lanes. There are many lines that could be captured in the image so machine learning models could be used to continually learn about what lines are road lanes. This idea stemmed from work shown on this website: https://medium.com/@cacheop/advanced-lane-detection-for-autonomous-cars-bff5390a360f.  (Kazuki)

Step 6: Object Detection
For object detection, we will focus on traffic signal detection using HSV color classification, and human face detection using haar cascades.For traffic signals, you should take only the top half of the image since spot lights are in the top. Use a spotlight detection algorithm to determine brighter areas of the image surrounded by darker areas, since this is what traffic signals have in images. However, this will give us too many options within the image, such glares on a car, etc. We need to prune this down to only traffic lights. We can do this by using image segmentation, to split the image into foreground and background, since a lot of the extraneous options would be in the background. We can use the watershed algorithm to segment the image in such a way that it is isolated to just the traffic signals. This will give us a much more limited number of possible areas for the traffic signals, which we can pass into a classifier to determine whether or not the options are actually traffic signals. Similar methods can be used for human detection. (Rishav and Kazuki)

Derived from the algorithm described here: https://medium.com/@kenan.r.alkiek/https-medium-com-kenan-r-alkiek-traffic-light-recognition-505d6ab913b1

Step 7: All the sensor information is then mapped to the action space. For example when a stop sign is detected, the value will be mapped to the break so that the vehicle will stop moving after the object is detected. (Kazuki)

# 6. Evaluation/Result: 
For evaluation we are going to be building a track for the RC car. This is going to include different things like traffic signals using leds, stop signs and miniature human toys. We are then going to test the car if it stop automatically at the correct times. Quantitative results can be measured through sensor data delivered through ROS messages. For example, if the agent is in motion and the camera recognized a red stop sign, the ROS node should publish values to the break telling it to stop. We can check the validity of our vision program by comparing it to state of the art deep learning detectors such as darknet YOLO.
