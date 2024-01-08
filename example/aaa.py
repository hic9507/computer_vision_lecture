import cv2
import numpy as np
# import matplotlib.pyplot as plt





# img = cv2.imread('stitching/boat1.jpg')
# corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),2,3,0.04)


# harris = np.copy(img)
# Canny = np.copy(img)


# corners = cv2.dilate(corners, None)

# harris[corners>0.1*corners.max()] = [0,0,255]

# corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# harris = np.hstack((harris, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)))
# Canny = cv2.Canny(Canny, 50, 200)


# cv2.imshow('Harris', harris)
# cv2.imshow('Canny', Canny)

# cv2.waitKey()



#---------------------------------------------------------------



# img_0 = cv2.imread('stitching/newspaper2.jpg')
# img_1 = cv2.imread('stitching/newspaper1.jpg')



# gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
# gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)


# detector = cv2.xfeatures2d.SURF_create()
# kps0, fea0 = detector.detectAndCompute(gray_0, None)
# kps1, fea1 = detector.detectAndCompute(gray_1, None)
# matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
# matches = matcher.match(fea0, fea1)

# pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
# pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
# H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

# dbg_img = cv2.drawMatches(img_0,kps0, img_1, kps1,[m for i,m in enumerate(matches) if mask[i]], None)
# cv2.imshow('surf',dbg_img[:,:,[2,1,0]])
# cv2.waitKey()





# detector = cv2.xfeatures2d.SIFT_create(50)
# kps0, fea0 = detector.detectAndCompute(img_0, None)
# kps1, fea1 = detector.detectAndCompute(img_1, None)
# matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
# matches = matcher.match(fea0, fea1)

# pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
# pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
# H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)


# dbg_img = cv2.drawMatches(img_0,kps0, img_1, kps1,[m for i,m in enumerate(matches) if mask[i]], None)
# cv2.imshow('sift',dbg_img[:,:,[2,1,0]])
# cv2.waitKey()


# detector = cv2.ORB_create(100)
# kps0, fea0 = detector.detectAndCompute(img_0, None)
# kps1, fea1 = detector.detectAndCompute(img_1, None)
# matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
# matches = matcher.match(fea0, fea1)

# pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
# pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)


# H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)



# dbg_img = cv2.drawMatches(img_0,kps0, img_1, kps1,[m for i,m in enumerate(matches) if mask[i]], None)
# cv2.imshow('orb',dbg_img[:,:,[2,1,0]])
# cv2.waitKey()




imgL = cv2.imread('stitching/newspaper2.jpg')
imgR = cv2.imread('stitching/newspaper1.jpg')


hL, wL = imgL.shape[:2]
hR, wR = imgR.shape[:2]


grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


descriptor = cv2.xfeatures2d.SIFT_create()
# descriptor = cv2.ORB_create(100)

kpsL, featuresL = descriptor.detectAndCompute(imgL, None)
kpsR, featuresR = descriptor.detectAndCompute(imgR, None)

imgL_draw = cv2.drawKeypoints(imgL, kpsL, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgR_draw = cv2.drawKeypoints(imgR, kpsR, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.knnMatch(featuresR, featuresL, 2)


good_matches = []

for m in matches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches.append((m[0].trainIdx, m[0].queryIdx))



if len(good_matches)>4:
    ptsL = np.float32([kpsL[i].pt for (i,_) in good_matches])
    ptsR = np.float32([kpsR[i].pt for (_,i) in good_matches])
    matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)

    panorama = cv2.warpPerspective(imgR, matrix, (wR+wL, hR))
    panorama[0:hL, 0:wL] = imgL
else:
    panorama = imgL

panorama =  cv2.resize(panorama, (1980, 1080))

cv2.imshow('panorama', panorama)
cv2.waitKey()




#---------------------------------------------------------------------------------

# import cv2

# img = []

# img.append(cv2.imread('stitching/boat1.jpg', cv2.IMREAD_COLOR))
# img.append(cv2.imread('stitching/boat2.jpg', cv2.IMREAD_COLOR))
# img.append(cv2.imread('stitching/boat3.jpg', cv2.IMREAD_COLOR))
# img.append(cv2.imread('stitching/boat4.jpg', cv2.IMREAD_COLOR))


# stitcher = cv2.createStitcher()
# ret, pano = stitcher.stitch(img)


# if ret == cv2.STITCHER_OK:
#     pano = cv2.resize(pano, dsize=(0,0), fx=0.2, fy=0.2)
#     cv2.imshow('panorama', pano)
#     cv2.waitKey()


#     cv2.destroyAllWindows()
# else:
#     print('Error during stiching')






# #--------------------------------



# img_0 = cv2.imread('stitching/dog_a.jpg')
# img_1 = cv2.imread('stitching/dog_b.jpg')


# gray0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
# gray1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

# pts0 = cv2.goodFeaturesToTrack(gray0, 500, 0.05, 10)
# pts1 = cv2.goodFeaturesToTrack(gray0, 500, 0.05, 10)


# for i in pts0:
#     cv2.circle(gray0, tuple(i[0]), 3, (0, 0, 255), 2)

# for i in pts1:
#     cv2.circle(gray1, tuple(i[0]), 3, (0, 0, 255), 2)

# images = []

# images.append(img_0)
# images.append(img_1)


# prev_pts = None
# prev_gray_frame = None
# tracks = None


# for img in images:

#     gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#     if prev_pts is not None:
#         pts, status, errors = cv2.calcOpticalFlowPyrLK(
#             prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5,
#             criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#         good_pts = pts[status==1]

#         if tracks is None:
#             tracks = good_pts
#         else:
#             tracks = np.vstack((tracks, good_pts))
#         for p in tracks:
#             cv2.circle(img, (p[0], p[1]),3 ,(0,255,0), -1)

#     else:
#         pts = cv2.goodFeaturesToTrack(gray_frame, 500, 0.05, 10)
#         pts = pts.reshape(-1,1,2)

#     prev_pts = pts
#     prev_gray_frame = gray_frame

#     cv2.imshow('frame', img)
#     key = cv2.waitKey() & 0xff
#     if key == 27: break
#     if key == ord('c'):
#         tracks = None
#         prev_pts = None

# cv2.destroyAllWindows()

