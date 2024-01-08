##########2번
import random
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

########################################################################### 1번 ########################################################################
# Canny Edge
# img0 = cv2.imread('./stitching/boat1.jpg', cv2.IMREAD_GRAYSCALE)
# img1 = cv2.imread('./stitching/budapest1.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_GRAYSCALE)
#
# edge0 = cv2.Canny(img0, 200, 100)
# edge1 = cv2.Canny(img1, 200, 100)
# edge2 = cv2.Canny(img2, 200, 100)
# edge3 = cv2.Canny(img3, 200, 100)
#
# cv2.imshow('edge0', edge0)
# cv2.imshow('edge1', edge1)
# cv2.imshow('edge2', edge2)
# cv2.imshow('edge3', edge3)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # Harris Corner
# img0 = cv2.imread('./stitching/boat1.jpg', cv2.IMREAD_COLOR)
# img1 = cv2.imread('./stitching/budapest1.jpg', cv2.IMREAD_COLOR)
# img2 = cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_COLOR)
# img3 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_COLOR)
#
# corners0 = cv2.cornerHarris(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
# corners1 = cv2.cornerHarris(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
# corners2 = cv2.cornerHarris(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
# corners3 = cv2.cornerHarris(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
#
# corners0 = cv2.dilate(corners0, None)
# corners1 = cv2.dilate(corners1, None)
# corners2 = cv2.dilate(corners2, None)
# corners3 = cv2.dilate(corners3, None)
#
# show_img0 = np.copy(img0)
# show_img1 = np.copy(img1)
# show_img2 = np.copy(img2)
# show_img3 = np.copy(img3)
#
# show_img0[corners0 > 0.1 * corners0.max()] = [0, 0, 255]
# show_img1[corners1 > 0.1 * corners1.max()] = [0, 0, 255]
# show_img2[corners2 > 0.1 * corners2.max()] = [0, 0, 255]
# show_img3[corners3 > 0.1 * corners3.max()] = [0, 0, 255]
#
# corners0 = cv2.normalize(corners0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# corners1 = cv2.normalize(corners1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# corners2 = cv2.normalize(corners2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# corners3 = cv2.normalize(corners3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#
# show_img0 = np.hstack((show_img0, cv2.cvtColor(corners0, cv2.COLOR_GRAY2BGR)))
# show_img1 = np.hstack((show_img1, cv2.cvtColor(corners1, cv2.COLOR_GRAY2BGR)))
# show_img2 = np.hstack((show_img2, cv2.cvtColor(corners2, cv2.COLOR_GRAY2BGR)))
# show_img3 = np.hstack((show_img3, cv2.cvtColor(corners3, cv2.COLOR_GRAY2BGR)))
#
# cv2.imshow('Harris corner detector_boat1', show_img0)
# cv2.imshow('Harris corner detector_budapest1', show_img1)
# cv2.imshow('Harris corner detector_newspaper1', show_img2)
# cv2.imshow('Harris corner detector_s1', show_img3)
#
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()

####################################################################### 2번 #####################################################################

# img0 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_COLOR)
# img1 = cv2.imread('./stitching/s2.jpg', cv2.IMREAD_COLOR)
# # img0 = cv2.resize(img0, (1000, 800))
# img1 = cv2.resize(img1, (1246, 700))
# imgs_list = [img0, img1]
#
#
# # SIFT
# detector_sift = cv2.xfeatures2d.SIFT_create(50)
# show_img = np.copy(imgs_list)
#
# for i in range(len(show_img)):
#     keypoints, descriptors = detector_sift.detectAndCompute(show_img[i], None)
#     show_img[i] = cv2.drawKeypoints(show_img[i], keypoints, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('SIFT keypoints0', (show_img[0]))
# cv2.imshow('SIFT keypoints1', (show_img[1]))
# cv2.imshow('SIFT', (np.hstack(show_img)))
# cv2.waitKey()
#
# keypoints0, descriptors0 = detector_sift.detectAndCompute(img0, None)
# keypoints1, descriptors1 = detector_sift.detectAndCompute(img1, None)
#
# kp_list = [keypoints0, keypoints1]
# desc_list = [descriptors0, descriptors1]
#
# sum = kp_list + desc_list
#
# # show_img = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # for kp in kp_list:
# #     kp.size = 100*random.random()
# #     kp.angle = 360*random.random()
# #
# # matches = []
# # for i in range(len(kp_list)):
# #     matches.append(cv2.DMatch(i, i, 1))
# #
# # # SURF
# detector_surf = cv2.xfeatures2d.SURF_create(1000, 3, True, True)
# show_img = np.copy(imgs_list)
#
# for i in range(len(show_img)):
#     keypoints, descriptors = detector_surf.detectAndCompute(show_img[i], None)
#
#     show_img[i] = cv2.drawKeypoints(show_img[i], keypoints, None, (0, 255, 0),
#                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('SURF keypoints', np.hstack(show_img))
# cv2.waitKey()
#
# # # ORB
# detector_orb = cv2.ORB_create()
# show_img = np.copy(imgs_list)
#
# for i in range(len(imgs_list)):
#     keypoints, descriptors = detector_orb.detectAndCompute(show_img[i], None)
#
#     show_img[i] = cv2.drawKeypoints(show_img[i], keypoints, None, (255, 0, 0),
#                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('ORB keypoints', np.hstack(show_img))
# cv2.waitKey()
# cv2.destroyAllWindows()

########################################################### 3번 #################################################################
# images = []
# images.append(cv2.imread('./stitching/boat1.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('./stitching/boat2.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('./stitching/boat3.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('./stitching/boat4.jpg', cv2.IMREAD_COLOR))
#
# stitcher = cv2.createStitcher()
# ret, pano = stitcher.stitch(images)
#
# if ret == cv2.STITCHER_OK:
#     pano = cv2.resize(pano, dsize=(0, 0), fx=0.2, fy=0.2)
#     cv2.imshow('panorama', pano)
#     cv2.waitKey()
#
#     cv2.destroyAllWindows()
# else:
#     print('Error during stitching')

############################################################ 4번 ################################################################
# prev_pts =None
# prev_gray_frame=None
# tracks = None
#
# for i in os.listdir('./stitching/dog/'):
#     path = './stitching/dog/' + i
#
#     #Good Features to tracking
#     images = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     corners = cv2.goodFeaturesToTrack(images, 100, 0.05, 10)
#
#     for c in corners:
#         x, y = c[0]
#         cv2.circle(images, (x, y), 5, 255, -1)
#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(images, cmap='gray')
#     # plt.tight_layout()
#     # plt.show()
#
#         # 루카스 카나데
#         retval, frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         if not retval: break
#         gray_frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
#
#         if prev_pts is not None:
#             pts, status, errors = cv2.calcOpticalFlowPyrLK(
#                 prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5,
#                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#             good_pts = pts[status == 1]
#             if tracks is None: tracks = good_pts
#             else: tracks = np.vstack((tracks, good_pts))
#             for p in tracks:
#                 cv2.circle(frame, (p[0], p[1]), 3, (0, 255, 0), -1)
#         else:
#             pts = cv2.goodFeaturesToTrack(gray_frame, 500, 0.05, 10)
#             pts = pts.reshape(-1, 1, 2)
#         prev_pts = pts
#         prev_gray_frame = gray_frame
#
#         cv2.imshow('frame', frame)
#         key = cv2.waitKey() & 0xff
#         if key == 27: break
#         if key == ord('c'):
#             tracks = None
#             prev_pts = None
# cv2.destroyAllWindows()


# for i in range(len(show_img)):
#     keypoints, descriptors = detector_surf.detectAndCompute(show_img[i], None)
#
#     show_img[i] = cv2.drawKeypoints(show_img[i], keypoints, None, (0, 255, 0),
#                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# image = []
# image.append(cv2.imread('C:/Users/user/Desktop/computer_vision/example/stitching/s1.jpg'))
# image.append(cv2.imread('C:/Users/user/Desktop/computer_vision/example/stitching/s2.jpg'))
# # SIFT, SURF, ORB를 추출한 후 매칭 및 RANSAC을 통해서 두 장의 영상간의 homography를 계산하고, 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping 하는 코드를 작성
#
# surf = cv2.xfeatures2d.SURF_create(10000)
# surf.setExtended(True)
# surf.setNOctaves(3)
# surf.setNOctaveLayers(10)
# surf.setUpright(False)
#
# keyPoints, descriptors = surf.detectAndCompute(image[0], None)
# keyPoints2, descriptors2 = surf.detectAndCompute(image[1], None)
#
# show_img = cv2.drawKeypoints(image[0], keyPoints, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img2 = cv2.drawKeypoints(image[1], keyPoints2, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('surf', show_img)
# cv2.imshow('surf2', show_img2)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# orb = cv2.ORB_create()
# orb.setMaxFeatures(200)
# keyPoints = orb.detect(image[0], None)
# keyPoints2 = orb.detect(image[1], None)
# keyPoints, descriptors = orb.compute(image[0], keyPoints)
# keyPoints2, descriptors2 = orb.compute(image[1], keyPoints2)
# show_img = cv2.drawKeypoints(image[0], keyPoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img2 = cv2.drawKeypoints(image[1], keyPoints2, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('orb', show_img)
# cv2.imshow('orb2', show_img2)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # image[1] = cv2.resize(image[1], None, fx=0.75, fy=0.75)
# # image[1] = np.pad(image[1],((64,)*2,(64,)*2,(0,)*2,), 'constant', constant_values=0)
# detector = cv2.xfeatures2d.SIFT_create(50)
# for i in range(len(image)):
#     keypoints, descriptors = detector.detectAndCompute(image[i], None)
#     image[i] = cv2.drawKeypoints(image[i], keypoints, None, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# cv2.imshow('shift', np.hstack(image))
# cv2.waitKey()
# cv2.destroyAllWindows()

images = []
images.append(cv2.imread('./stitching/dog_a.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('./stitching/dog_b.jpg', cv2.IMREAD_COLOR))

#-------------4-1
tracks = None

pts = cv2.goodFeaturesToTrack(cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY), 100, 0.05, 10)
pts = pts.reshape(-1,1,2)
prev_pts=pts

prev_gray_frame = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
gray_frame = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
pts, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
good_pts = pts[status==1]
if tracks is None: tracks=good_pts
else: tracks = np.vstack((tracks, good_pts))
for p in tracks:
    cv2.circle(images[0], (p[0],p[1]),3,(0,255,0),-1)
cv2.imshow('lucas-kanade', images[0])
cv2.waitKey()
cv2.destroyAllWindows()