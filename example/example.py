import random
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

########################################################################### 1번 시작 ########################################################################
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

####################################################################### 1번 끝 #####################################################################

####################################################################### 2번 시작 #####################################################################
# img0 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_COLOR)
# img1 = cv2.imread('./stitching/s2.jpg', cv2.IMREAD_COLOR)
# # img0 = cv2.resize(img0, (1000, 800))
# img1 = cv2.resize(img1, (1246, 700))
# imgs_list = [img0, img1]
#
# # SIFT
# detector_sift = cv2.xfeatures2d.SIFT_create(50)
# show_img = np.copy(imgs_list)
#
# # for i in range(len(show_img)):
# keypoints0, descriptors0 = detector_sift.detectAndCompute(show_img[0], None)
# keypoints1, descriptors1 = detector_sift.detectAndCompute(show_img[1], None)
#
# matcher_sift = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
# matches_sift = matcher_sift.match(descriptors0, descriptors1)
#
# pts0 = np.float32([keypoints0[m.queryIdx].pt for m in matches_sift]).reshape(-1, 2)
# pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches_sift]).reshape(-1, 2)
#
# H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
#
# show_img[0] = cv2.drawKeypoints(show_img[0], keypoints0, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img[1] = cv2.drawKeypoints(show_img[1], keypoints1, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# dbg_img = cv2.drawMatches(show_img[0], keypoints0, show_img[1], keypoints1,[m for i,m in enumerate(matches_sift) if mask[i]], None)
#
#
# cv2.imshow('SIFT keypoints0', (show_img[0]))
# cv2.imshow('SIFT keypoints1', (show_img[1]))
# cv2.imshow('SIFT', (np.hstack(show_img)))
# cv2.imshow('sift', dbg_img[:, :, [2, 1, 0]])
# cv2.waitKey()
#
# # # SURF
# detector_surf = cv2.xfeatures2d.SURF_create(10000)
# detector_surf.setExtended(True)
# detector_surf.setNOctaves(3)
# detector_surf.setNOctaveLayers(10)
# detector_surf.setUpright(False)
#
# show_img = np.copy(imgs_list)
#
# keypoints0, descriptors0 = detector_surf.detectAndCompute(show_img[0], None)
# keypoints1, descriptors1 = detector_surf.detectAndCompute(show_img[1], None)
#
# matcher_surf = cv2.BFMatcher_create(cv2.NORM_L2, False)
# matches_surf = matcher_surf.match(descriptors0, descriptors1)
#
# pts0 = np.float32([keypoints0[m.queryIdx].pt for m in matches_surf]).reshape(-1, 2)
# pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches_surf]).reshape(-1, 2)
#
# H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
#
# show_img0 = cv2.drawKeypoints(show_img[0], keypoints0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img1 = cv2.drawKeypoints(show_img[1], keypoints1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# show_img = [show_img0, show_img1]
#
# dbg_img = cv2.drawMatches(show_img0, keypoints0, show_img1, keypoints1, [m for i, m in enumerate(matches_surf) if mask[i]], None)
#
#
# cv2.imshow('SURF keypoints0', show_img0)
# cv2.imshow('SURF keypoints1', show_img1)
# cv2.imshow('SURF', np.hstack(show_img))
# cv2.imshow('surf', dbg_img[:, :, [2, 1, 0]])
# cv2.waitKey()
#
# # # ORB
# detector_orb = cv2.ORB_create()
# detector_orb.setMaxFeatures(100)
#
# show_img = np.copy(imgs_list)
#
# keypoints2 = detector_orb.detect(show_img[0], None)
# keypoints3 = detector_orb.detect(show_img[1], None)
# keypoints2, descriptors2 = detector_orb.compute(show_img[0], keypoints2)
# keypoints3, descriptors3 = detector_orb.compute(show_img[1], keypoints3)
#
# kps4, fea0 = detector_orb.detectAndCompute(show_img[0], None)
# kps5, fea1 = detector_orb.detectAndCompute(show_img[1], None)
#
# matcher_orb = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
# matches_orb = matcher_orb.match(fea0, fea1)
#
# pts0 = np.float32([kps4[m.queryIdx].pt for m in matches_orb]).reshape(-1, 2)
# pts1 = np.float32([kps5[m.queryIdx].pt for m in matches_orb]).reshape(-1, 2)
#
# H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
#
# show_img2 = cv2.drawKeypoints(show_img[0], keypoints2, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img3 = cv2.drawKeypoints(show_img[1], keypoints3, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# show_img = [show_img2, show_img3]
#
# dbg_img = cv2.drawMatches(show_img[0], kps4, show_img[1], kps5, [m for i, m in enumerate(matches_orb) if mask[i]], None)
#
# cv2.imshow('ORB keypoints0', show_img2)
# cv2.imshow('ORB keypoints1', show_img3)
# cv2.imshow('ORB ', np.hstack(show_img))
# cv2.imshow('orb ', dbg_img[:, :, [2, 1, 0]])
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# show_img = np.copy(imgs_list)
# hL, wL = show_img[0].shape[:2]
# hR, wR = show_img[1].shape[:2]
#
# gray0 = cv2.cvtColor(show_img[0], cv2.COLOR_BGR2GRAY)
# gray1 = cv2.cvtColor(show_img[1], cv2.COLOR_BGR2GRAY)
#
# # SIFT 디스크럽터 추출기 생성
# detector_surf = cv2.xfeatures2d.SURF_create(10000)
#
# # 각 영상에 대해 키 포인트와 디스크럽터 추출
# keypoints0, descriptors0 = detector_sift.detectAndCompute(show_img[0], None)
# keypoints1, descriptors1 = detector_sift.detectAndCompute(show_img[1], None)
#
# # assign
# # show_img[0] = cv2.drawKeypoints(show_img[0], keypoints0, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # show_img[1] = cv2.drawKeypoints(show_img[1], keypoints1, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img0 = cv2.drawKeypoints(show_img[0], keypoints0, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img1 = cv2.drawKeypoints(show_img[1], keypoints1, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # matcher_sift = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
# # matches_sift = matcher_sift.match(descriptors0, descriptors1)
# matcher_sift = cv2.DescriptorMatcher_create('BruteForce')
# matches_sift = matcher_sift.knnMatch(descriptors1, descriptors0, 2)
#
# good_matches = []
# for m in matches_sift:
#     if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
#         good_matches.append((m[0].trainIdx, m[0].queryIdx))
#
# if len(good_matches) > 4:
#     ptsL = np.float32([keypoints0[i].pt for (i, _) in good_matches])
#     ptsR = np.float32([keypoints1[i].pt for (_, i) in good_matches])
#     matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)
#     panorama = cv2.warpPerspective(show_img[1], matrix, (wR + wL, hR))
#     panorama[0:hL, 0:wL] = show_img[0]
# else:
#     panorama = show_img[0]
#
# cv2.imshow('panorama', panorama)
# cv2.waitKey()

########################################################### 2번 끝 #####################################################################


########################################################### 2번 쓰잘데기 없는거 ########################################################
# cv2.imshow('1', show_img[0])
# cv2.imshow('2', show_img[1])
# cv2.waitKey()

# pts0 = np.float32([keypoints0[m.queryIdx].pt for m in matcher_orb]).reshape(-1, 2)
# pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matcher_orb]).reshape(-1, 2)

# width = show_img2.shape[0] + show_img3[1]
# height = show_img2.shape[1] + show_img3[0]

# print(pts0)

# 여기서부터 2번 끝까지 다 주석처리했음 풀어도 될듯?
# result = cv2.warpPerspective(show_img2, H, (width, height))
# result[0:show_img3[0], 0:show_img3[1]] = show_img3
# plt.figure(figsize=(20,10))
# plt.imshow(result)
#
# plt.axis('off')
# plt.show()
#
# # perspec = cv2.getPerspectiveTransform(pts0, pts1)
# # unwarped = cv2.warpPerspective(show_img[0], perspec, (pts0, pts1))
# #
# # cv2.imshow('11221', np.hstack(show_img[0],unwarped))
#
#
# plt.figure()
# plt.subplot(211)
# plt.axis('off')
# plt.title('all matches')
# # dbg_img = cv2.drawMatches(show_img[0], kps4, show_img[1], kps5, matches, None)
# dbg_img = cv2.drawMatches(show_img[0], keypoints2, show_img[1], keypoints3, matches, None)
# plt.imshow(dbg_img[:,:,[2,1,0]])
# plt.subplot(212)
# plt.axis('off')
# plt.title('filtered matches')
# # dbg_img = cv2.drawMatches(show_img[0], kps4, show_img[1], kps5, [m for i,m in enumerate(matches) if mask[i]], None)
# dbg_img = cv2.drawMatches(show_img[0], keypoints2, show_img[1], keypoints3, [m for i,m in enumerate(matches) if mask[i]], None)
# plt.imshow(dbg_img[:,:,[2,1,0]])
# plt.tight_layout()
# plt.show()
#
# hL, wL = show_img[0].shape[:2]
# hR, wR = show_img[1].shape[:2]
# print(hL)
# print(hR)
#
# show_img = np.copy(imgs_list)
#
# good_matches = []
# panorama = cv2.warpPerspective(show_img[1], H, (wR + wL, hR))
# panorama[0:hL, 0:wL] = show_img[0]
# # for m in matches:
# #     if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
# #         good_matches.append((m[0].trainIdx, m[0].queryIdx))
# # if len(good_matches) > 4:
# #     ptsL = np.float32(kps4[i].pt for (i, _) in good_matches)
# #     ptsR = np.float32(kps4[i].pt for (i, _) in good_matches)
# #     matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)
# #
# # else:
# #     panorama = show_img[0]
# cv2.imshow('1', panorama)
# cv2.waitKey()
#     # kps4, fea0 = detector_orb.detectAndCompute(show_img[0], None)
#     # kps5, fea1 = detector_orb.detectAndCompute(show_img[1], None)
#
# # select_points = [kps4, kps5]
# # src_pts = select_points(show_img, 4)
# # dst_pts = np.array([[0, 240], [0, 0], [240, 0], [240, 240]], dtype=np.float32)
# # perspevtive_m = cv2.getPerspectiveTransform(src_pts, dst_pts)
# # unwarped_img = cv2.warpPerspective(show_img3, unwarped_img)
#
# # print(len(show_img))
########################################################### 2번 쓰잘데기 없는거 끝 ########################################################

########################################################### 3번 시작 #################################################################
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

####################################################################### 3번 끝 #####################################################################

############################################################ 4번 시작 ################################################################
prev_pts =None
prev_gray_frame=None
tracks = None

images = []
images.append(cv2.imread('./stitching/dog_a.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('./stitching/dog_b.jpg', cv2.IMREAD_COLOR))

###################################################################### 4-1 시작 ###############################################################
# while True:
# ####### Good Features to tracking
#     gray_frame0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
#     gray_frame = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY) #####
#
#     corners = cv2.goodFeaturesToTrack(gray_frame, 100, 0.05, 10)
#     pts = corners.reshape(-1, 1, 2)
#     prev_pts = pts
#     # prev_gray_frame = gray_frame
#     prev_gray_frame = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)#####
#     ###### Lucas-Kanade 알고리즘
#     pts, status, erros = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#     good_pts = pts[status == 1]
#     if tracks is None:
#         tracks = good_pts
#     else: tracks = np.vstack((tracks, good_pts))
#     for p in tracks:
#         cv2.circle(images[0], (p[0], p[1]), 3, (255, 255, 255), -1)
#
#     cv2.imshow('frame', images[0])
#     key = cv2.waitKey() & 0xff
#     if key == 27:
#         break
#     if key == ord('c'):
#         tracks = None
#         prev_pts = None
###################################################################### 4-1 끝 ###############################################################
#ㅇ########################## 4-1 쓰잘데기 없는거 ########################
# for c in corners:
#     x, y = c[0]
#     cv2.circle(images, (x, y), 5, 255, -1)
# plt.figure(figsize=(10, 10))
# plt.imshow(images, cmap='gray')
# plt.tight_layout()
# plt.show()

    # 루카스 카나데
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
#ㅇ########################## 4-1 쓰잘데기 없는거 ########################

###################################################################### 4-2 시작 ###############################################################
show_img = np.copy(images)

def display_flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10*delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1],
                            (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)
    
    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1,
                                  cv2.NORM_MINMAX)
    
    cv2.imshow('optical flow', img)
    cv2.imshow('optical flow magnitude', norm_opt_flow)
    cv2.waitKey()
    # k = cv2.waitKey(1)
    #
    # if k == 27:
    #     return 1
    # else:
    #     return 0

prev_frame = cv2.cvtColor(show_img[0], cv2.COLOR_BGR2GRAY) #prev_frame
prev_frame = cv2.resize(prev_frame, (0, 0), None, 0.5, 0.5)
init_flow = True

gray_frame = cv2.cvtColor(show_img[1], cv2.COLOR_BGR2GRAY)  #frame
gray_frame = cv2.resize(gray_frame, (0, 0), None, 0.5, 0.5)

opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

if init_flow:
    opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    init_flow = False

else:
    opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, opt_flow, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_USE_INITIAL_FLOW)
prev_frame = np.copy(gray_frame)

if display_flow(gray_frame, opt_flow):
    pass
# display_flow(gray_frame, opt_flow)

##########################################################################################################
prev_frame = cv2.cvtColor(show_img[0], cv2.COLOR_BGR2GRAY) #prev_frame
prev_frame = cv2.resize(prev_frame, (0, 0), None, 0.5, 0.5)
init_flow = True

gray_frame = cv2.cvtColor(show_img[1], cv2.COLOR_BGR2GRAY)  #frame
gray_frame = cv2.resize(gray_frame, (0, 0), None, 0.5, 0.5)

flow_DualTVL1 = cv2.createOptFlow_DualTVL1()

opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, None)

if init_flow:
    opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, None)
    init_flow = False

if not flow_DualTVL1.getUseInitialFlow():
    opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, None)
    flow_DualTVL1.setUseInitialFlow(True)
else:
    opt_flow = flow_DualTVL1.calc(prev_frame, gray_frame, opt_flow)

prev_frame = np.copy(gray_frame)

if display_flow(gray_frame, opt_flow):
    pass