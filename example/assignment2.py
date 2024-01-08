import cv2
import matplotlib.pyplot as plt
import numpy as np

img0 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('./stitching/s2.jpg', cv2.IMREAD_COLOR)

img1 = cv2.resize(img1, (1246, 700))
imgs_list = [img0, img1]

## SIFT
detector_sift = cv2.xfeatures2d.SIFT_create(50)
show_img = np.copy(imgs_list)


keypoints0, descriptors0 = detector_sift.detectAndCompute(show_img[0], None)
keypoints1, descriptors1 = detector_sift.detectAndCompute(show_img[1], None)

matcher_sift = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
matches_sift = matcher_sift.match(descriptors0, descriptors1)

pts0 = np.float32([keypoints0[m.queryIdx].pt for m in matches_sift]).reshape(-1, 2)
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches_sift]).reshape(-1, 2)

H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

show_img[0] = cv2.drawKeypoints(show_img[0], keypoints0, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_img[1] = cv2.drawKeypoints(show_img[1], keypoints1, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

dbg_img = cv2.drawMatches(show_img[0], keypoints0, show_img[1], keypoints1,[m for i,m in enumerate(matches_sift) if mask[i]], None)

cv2.imshow('SIFT keypoints0', (show_img[0]))
cv2.imshow('SIFT keypoints1', (show_img[1]))
cv2.imshow('SIFT', (np.hstack(show_img)))
cv2.imshow('sift', dbg_img[:, :, [0, 1, 2]])
cv2.waitKey()


## SURF
detector_surf = cv2.xfeatures2d.SURF_create(10000)
detector_surf.setExtended(True)
detector_surf.setNOctaves(3)
detector_surf.setNOctaveLayers(10)
detector_surf.setUpright(False)

show_img = np.copy(imgs_list)

keypoints0, descriptors0 = detector_surf.detectAndCompute(show_img[0], None)
keypoints1, descriptors1 = detector_surf.detectAndCompute(show_img[1], None)

matcher_surf = cv2.BFMatcher_create(cv2.NORM_L2, False)
matches_surf = matcher_surf.match(descriptors0, descriptors1)

pts0 = np.float32([keypoints0[m.queryIdx].pt for m in matches_surf]).reshape(-1, 2)
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches_surf]).reshape(-1, 2)

H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

show_img0 = cv2.drawKeypoints(show_img[0], keypoints0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_img1 = cv2.drawKeypoints(show_img[1], keypoints1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_img = [show_img0, show_img1]

dbg_img = cv2.drawMatches(show_img0, keypoints0, show_img1, keypoints1, [m for i, m in enumerate(matches_surf) if mask[i]], None)

cv2.imshow('SURF keypoints0', show_img0)
cv2.imshow('SURF keypoints1', show_img1)
cv2.imshow('SURF', np.hstack(show_img))
cv2.imshow('surf', dbg_img[:, :, [0, 1, 2]])
cv2.waitKey()


## ORB
detector_orb = cv2.ORB_create()
detector_orb.setMaxFeatures(100)

show_img = np.copy(imgs_list)

keypoints2 = detector_orb.detect(show_img[0], None)
keypoints3 = detector_orb.detect(show_img[1], None)
keypoints2, descriptors2 = detector_orb.compute(show_img[0], keypoints2)
keypoints3, descriptors3 = detector_orb.compute(show_img[1], keypoints3)

kps4, fea0 = detector_orb.detectAndCompute(show_img[0], None)
kps5, fea1 = detector_orb.detectAndCompute(show_img[1], None)

matcher_orb = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matches_orb = matcher_orb.match(fea0, fea1)

pts0 = np.float32([kps4[m.queryIdx].pt for m in matches_orb]).reshape(-1, 2)
pts1 = np.float32([kps5[m.queryIdx].pt for m in matches_orb]).reshape(-1, 2)

H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)

show_img2 = cv2.drawKeypoints(show_img[0], keypoints2, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_img3 = cv2.drawKeypoints(show_img[1], keypoints3, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_img = [show_img2, show_img3]

dbg_img = cv2.drawMatches(show_img[0], kps4, show_img[1], kps5, [m for i, m in enumerate(matches_orb) if mask[i]], None)

cv2.imshow('ORB keypoints0', show_img2)
cv2.imshow('ORB keypoints1', show_img3)
cv2.imshow('ORB ', np.hstack(show_img))
cv2.imshow('orb ', dbg_img[:, :, [0, 1, 2]])
cv2.waitKey()
cv2.destroyAllWindows()

show_img = np.copy(imgs_list)
hL, wL = show_img[0].shape[:2]
hR, wR = show_img[1].shape[:2]

gray0 = cv2.cvtColor(show_img[0], cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(show_img[1], cv2.COLOR_BGR2GRAY)

# SIFT 디스크럽터 추출기 생성
detector_surf = cv2.xfeatures2d.SURF_create(10000)

# 각 영상에 대해 키 포인트와 디스크럽터 추출
keypoints0, descriptors0 = detector_sift.detectAndCompute(show_img[0], None)
keypoints1, descriptors1 = detector_sift.detectAndCompute(show_img[1], None)

## assign
show_img0 = cv2.drawKeypoints(show_img[0], keypoints0, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_img1 = cv2.drawKeypoints(show_img[1], keypoints1, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


matcher_sift = cv2.DescriptorMatcher_create('BruteForce')
matches_sift = matcher_sift.knnMatch(descriptors1, descriptors0, 2)

matcher_surf = cv2.DescriptorMatcher_create('BruteForce')
matches_surf = matcher_sift.knnMatch(descriptors1, descriptors0, 2)

matcher_orb = cv2.DescriptorMatcher_create('BruteForce')
matches_orb = matcher_sift.knnMatch(descriptors1, descriptors0, 2)

### SIFT
good_matches = []
for m in matches_sift:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches.append((m[0].trainIdx, m[0].queryIdx))

if len(good_matches) > 4:
    ptsL = np.float32([keypoints0[i].pt for (i, _) in good_matches])
    ptsR = np.float32([keypoints1[i].pt for (_, i) in good_matches])
    matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)
    panorama_sift = cv2.warpPerspective(show_img[1], matrix, (wR + wL, hR))
    panorama_sift[0:hL, 0:wL] = show_img[0]
else:
    panorama_sift = show_img[0]


### SURF
good_matches = []
for m in matches_surf:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches.append((m[0].trainIdx, m[0].queryIdx))

if len(good_matches) > 4:
    ptsL = np.float32([keypoints0[i].pt for (i, _) in good_matches])
    ptsR = np.float32([keypoints1[i].pt for (_, i) in good_matches])
    matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)
    panorama_surf = cv2.warpPerspective(show_img[1], matrix, (wR + wL, hR))
    panorama_surf[0:hL, 0:wL] = show_img[0]
else:
    panorama_surf = show_img[0]

### ORB
good_matches = []
for m in matches_sift:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches.append((m[0].trainIdx, m[0].queryIdx))

if len(good_matches) > 4:
    ptsL = np.float32([keypoints0[i].pt for (i, _) in good_matches])
    ptsR = np.float32([keypoints1[i].pt for (_, i) in good_matches])
    matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)
    panorama_orb = cv2.warpPerspective(show_img[1], matrix, (wR + wL, hR))
    panorama_orb[0:hL, 0:wL] = show_img[0]
else:
    panorama_orb = show_img[0]

cv2.imshow('panorama_sift', panorama_sift)
cv2.imshow('panorama_surf', panorama_surf)
cv2.imshow('panorama_orb', panorama_orb)
cv2.waitKey()