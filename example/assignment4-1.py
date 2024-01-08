import cv2
import matplotlib.pyplot as plt
import numpy as np

prev_pts =None
prev_gray_frame=None
tracks = None

images = []
images.append(cv2.imread('./stitching/dog_a.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('./stitching/dog_b.jpg', cv2.IMREAD_COLOR))

###################################################################### 4-1 시작 ###############################################################

####### Good Features to tracking
gray_frame0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
gray_frame = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY) #####

corners = cv2.goodFeaturesToTrack(gray_frame, 100, 0.05, 10)
pts = corners.reshape(-1, 1, 2)
prev_pts = pts
# prev_gray_frame = gray_frame
prev_gray_frame = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)#####
###### Lucas-Kanade 알고리즘
pts, status, erros = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
good_pts = pts[status == 1]
if tracks is None:
    tracks = good_pts
else: tracks = np.vstack((tracks, good_pts))
for p in tracks:
    cv2.circle(images[0], (p[0], p[1]), 3, (255, 255, 255), -1)

cv2.imshow('frame', images[0])
key = cv2.waitKey() & 0xff
# if key == 27:
#     break
if key == ord('c'):
    tracks = None
    prev_pts = None
###################################################################### 4-1 끝 ###############################################################