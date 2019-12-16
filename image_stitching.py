#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from numpy.linalg import svd, inv

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
def auto_homography(Ia,Ib, homography_func=None, normalization_func=None, threshold=1, iteration=1000, thres_dist=0.75):
    '''
    Computes a homography that maps points from Ia to Ib
    Input: Ia and Ib are images
    Output: H is the homography
    '''
    Ia = Ia.copy()
    Ib = Ib.copy()
    
    if Ia.dtype == 'float32' and Ib.dtype == 'float32':
        Ia = (Ia*255).astype(np.uint8)
        Ib = (Ib*255).astype(np.uint8)
    
#     Ia[Ia>200] = 255
#     Ib[Ib>200] = 255
    Ia_gray = cv2.cvtColor(Ia,cv2.COLOR_BGR2GRAY)
    Ib_gray = cv2.cvtColor(Ib,cv2.COLOR_BGR2GRAY)

#     Ia_gray = Ia[:, :, 0]
#     Ib_gray = Ib[:, :, 0]
    
    #Ia_gray[Ia_gray>200] = 255
    #Ib_gray[Ib_gray> 200] = 255

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp_a, des_a = sift.detectAndCompute(Ia_gray,None)
    kp_b, des_b = sift.detectAndCompute(Ib_gray,None)    
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a,des_b, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < thres_dist*n.distance:
            good.append(m)
   
    numMatches = int(len(good))
    matches = good

    # Xa and Xb are 3xN matrices that contain homogeneous coordinates for the N
    # matching points for each image
    Xa = np.ones((3,numMatches))
    Xb = np.ones((3,numMatches))
    
    for idx, match_i in enumerate(matches):
        Xa[:,idx][0:2] = kp_a[match_i.queryIdx].pt
        Xb[:,idx][0:2] = kp_b[match_i.trainIdx].pt

    ## RANSAC
    niter = iteration
    best_score = 0
    H = 0
    for t in range(niter):
        # estimate homography
        subset = np.random.choice(numMatches, 15, replace=False)
        pts1 = Xa[:,subset]
        pts2 = Xb[:,subset]
        
        H_t = homography_func(pts1, pts2)#normalization_func) # edit helper code below (computeHomography)
        #_t = computeHomography(pts1, pts2)
        # score homography
        Xb_ = np.dot(H_t, Xa) # project points from first image to second using H
        du = Xb_[0,:]/Xb_[2,:] - Xb[0,:]/Xb[2,:]
        dv = Xb_[1,:]/Xb_[2,:] - Xb[1,:]/Xb[2,:]
        ok_t = np.sqrt(du**2 + dv**2) < threshold # you may need to play with this threshold
        score_t = sum(ok_t)

        if score_t > best_score:
            best_score = score_t
            H = H_t
            in_idx = ok_t
    
    print('best score: {:02f}'.format(best_score))

    # Optionally, you may want to re-estimate H based on inliers

    return H

def computeHomography(pts1, pts2):
    '''
    Compute homography that maps from pts1 to pts2 using least squares solver
     
    Input: pts1 and pts2 are 3xN matrices for N points in homogeneous
    coordinates. 
    
    Output: H is a 3x3 matrix, such that pts2~=H*pts1
    '''
    A = np.zeros((2*pts1.shape[1], 9))
    i = 0
    m = 0
    while i!=2*pts1.shape[1]:
        u = pts1[0, m]/pts1[2, m]
        v = pts1[1, m]/pts1[2, m]
        ud = pts2[0, m]/pts2[2, m]
        vd = pts2[1, m]/pts2[2, m]
        A[i][0] = -1*u
        A[i][1] = -1*v
        A[i][2] = -1
        A[i][6] = u*ud
        A[i][7] = v*ud
        A[i][8] = ud
        i=i+1
        A[i][3] = -1*u
        A[i][4] = -1*v
        A[i][5] = -1
        A[i][6] = u*vd
        A[i][7] = v*vd
        A[i][8] = vd
        i = i+1
        m = m+1
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    return vh[-1, :].reshape((3, 3))
    raise Exception("TODO in computeHomography() not implemented")
    
def getTranslation(H):
    im1_coordinates =np.array([[0, 0, 1], [360, 0, 1], [0, 480, 1], [360, 480, 1]])
    coord = H@im1_coordinates.T
    coord[0, :] = coord[0, :]/coord[2, :]
    coord[1, :] = coord[1, :]/coord[2, :]
    Ht = np.zeros((3, 3))
    Ht[0, 0] = Ht[1, 1] = Ht[2, 2] = 1
    Ht[0, 2] = -1*coord[0, 0]
    Ht[1, 2] = -1*coord[1, 0]
    return Ht

def video2imageFolder(input_file):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_file: Input video file.
        output_path: Output directorys.
    Output:
        None
    '''

    cap = cv2.VideoCapture()
    cap.open(input_file)

    if not cap.isOpened():
        print("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idx = 0
    frames = []
    while frame_idx < frame_count:
        ret, frame = cap.read()

        if not ret:
            print ("Failed to get the frame {}".format(frameId))
            continue
        frames.append(frame)
        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
    return frames
def imageFolder2mpeg(frames, output_path='./output_video.mpeg', fps=30.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_path: Input video file.
        output_path: Output directorys.
        fps: frames per second (default: 30).
    Output:
        None
    '''
    print(frames[0].shape)
    frame_Height, frame_Width = frames[0].shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MPG1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    frame_count = len(frames)

    frame_idx = 0

    while frame_idx < frame_count:

        video_writer.write(frames[frame_idx])
        frame_idx += 1

    video_writer.release()
    
def getVideo(vid1, vid2, vid3):
    vid1frames = video2imageFolder(vid1)
    vid2frames = video2imageFolder(vid2)
    vid3frames = video2imageFolder(vid3)
    return vid1frames, vid2frames, vid3frames

def getBestHomo(vid1frames, vid2frames, vid3frames, i, threshold=1, iteration=1000, thres_dist=0.75):
    
    H0 = auto_homography(vid1frames[i], vid2frames[i], computeHomography, threshold=threshold, iteration=iteration, thres_dist=thres_dist)
    H2 = auto_homography(vid3frames[i], vid2frames[i], computeHomography, threshold=threshold, iteration=iteration, thres_dist=thres_dist)
    T0 = getTranslation(H0)
    total_x = 1700 #int(480 + 2*abs(T0[0, 2]))
    total_y = 500 #int(360 + 50 + 2*abs(T0[1, 2]))
    img_warped0 = cv2.warpPerspective(vid1frames[i], T0.dot(H0), (total_x, total_y))
    img_warped2 = cv2.warpPerspective(vid3frames[i], T0.dot(H2), (total_x, total_y))
    img_warped1 = cv2.warpPerspective(vid2frames[i], T0, (total_x, total_y))
    final = (img_warped1 == 0)*img_warped0 + img_warped1 + (img_warped1 == 0)*img_warped2
    return final, H0, H2, T0

def applyHomo(vid1frames, vid2frames, vid3frames, H0, H2, T0):
    finalframes = []
    for i in range(len(vid1frames)):
        total_x = 1700 #int(480 + 2*abs(T0[0, 2]))
        total_y = 500 #int(360 + 50 + 2*abs(T0[1, 2]))
        img_warped0 = cv2.warpPerspective(vid1frames[i], T0.dot(H0), (total_x, total_y))
        img_warped2 = cv2.warpPerspective(vid3frames[i], T0.dot(H2), (total_x, total_y))
        img_warped1 = cv2.warpPerspective(vid2frames[i], T0, (total_x, total_y))
        final = (img_warped1 == 0)*img_warped0 + img_warped1 + (img_warped1 == 0)*img_warped2
        finalframes.append(final)
        cv2.imwrite('aut2/a{:04d}.jpg'.format(i), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    #output_p = './output_videoAUT.mpeg'
    #imageFolder2mpeg(finalframes, output_path='./output_videoAUT.mpeg', fps=30.0)
    return output_p
    
def getFinal(vid1frames, vid2frames, vid3frames):
    finalframes = []
    for i in range(len(vid1frames)):
        H0 = auto_homography(vid1frames[i] ,vid2frames[i], computeHomography)
        H2 = auto_homography(vid3frames[i] ,vid2frames[i], computeHomography)
        T0 = getTranslation(H0)
        total_x = 1700 #int(480 + 2*abs(T0[0, 2]))
        total_y = 500 #int(360 + 50 + 2*abs(T0[1, 2]))
        img_warped0 = cv2.warpPerspective(vid1frames[i], T0.dot(H0), (total_x, total_y))
        img_warped2 = cv2.warpPerspective(vid3frames[i], T0.dot(H2), (total_x, total_y))
        img_warped1 = cv2.warpPerspective(vid2frames[i], T0, (total_x, total_y))
        final = (img_warped1 == 0)*img_warped0 + img_warped1 + (img_warped1 == 0)*img_warped2
        finalframes.append(final)
        cv2.imwrite('aut/a{:04d}.jpg'.format(i), final)
    output_p = './output_videoAUT.mpeg'
    imageFolder2mpeg(finalframes, output_path='./output_videoAUT.mpeg', fps=30.0)
    return output_p


vid1 = './bag2_cam1.mp4'
vid2 = './bag2_cam2.mp4'
vid3 = './bag2_cam3.mp4'
a, b, c = getVideo(vid1, vid2, vid3)

len(a)

plt.imshow(a[600])

plt.imshow(b[600])

plt.imshow(c[600])


final, H0, H2, T0 = getBestHomo(a, b, c, i=0, threshold=10, iteration=2000, thres_dist=0.70)
plt.imshow(final)

o = applyHomo(a, b, c, H0, H2, T0)

import os
import cv2

import numpy as np
from math import floor
from numpy.linalg import svd, inv
def imageFolder2mpeg(input_path, output_path='./output_video.mpeg', fps=30.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_path: Input video file.
        output_path: Output directorys.
        fps: frames per second (default: 30).
    Output:
        None
    '''

    dir_frames = input_path
    files_info = os.scandir(dir_frames)

    file_names = [f.path for f in files_info if f.name.endswith(".jpg")]
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame_Height, frame_Width = cv2.imread(file_names[0]).shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MPG1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    frame_count = len(file_names)

    frame_idx = 0

    while frame_idx < frame_count:


        frame_i = cv2.imread(file_names[frame_idx])
        video_writer.write(frame_i)
        frame_idx += 1

    video_writer.release()
imageFolder2mpeg('./aut2', output_path='./output_video_aut2_new.mpeg', fps=30.0)





