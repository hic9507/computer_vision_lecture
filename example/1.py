import cv2
import numpy as np
image = []
image.append(cv2.imread('C:/Users/user/Desktop/computer_vision/example/stitching/s1.jpg'))
image.append(cv2.imread('C:/Users/user/Desktop/computer_vision/example/stitching/s2.jpg'))
#SIFT, SURF, ORB를 추출한 후 매칭 및 RANSAC을 통해서 두 장의 영상간의 homography를 계산하고, 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping 하는 코드를 작성

surf = cv2.xfeatures2d.SURF_create(10000)
surf.setExtended(True)
surf.setNOctaves(3)
surf.setNOctaveLayers(10)
surf.setUpright(False)

keyPoints, descriptors = surf.detectAndCompute(image[0], None)
keyPoints2, descriptors2 = surf.detectAndCompute(image[1], None)

show_img = cv2.drawKeypoints(image[0], keyPoints, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_img2 = cv2.drawKeypoints(image[1], keyPoints2, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('surf', show_img)
cv2.imshow('surf2', show_img2)
cv2.waitKey()
cv2.destroyAllWindows()

orb = cv2.ORB_create()
orb.setMaxFeatures(200)

keyPoints = orb.detect(image[0], None)
keyPoints2 = orb.detect(image[1], None)
keyPoints, descriptors = orb.compute(image[0], keyPoints)
keyPoints2, descriptors2 = orb.compute(image[1], keyPoints2)
show_img = cv2.drawKeypoints(image[0], keyPoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_img2 = cv2.drawKeypoints(image[1], keyPoints2, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('orb', show_img)
cv2.imshow('orb2', show_img2)
cv2.waitKey()
cv2.destroyAllWindows()

# image[1] = cv2.resize(image[1], None, fx=0.75, fy=0.75)
# image[1] = np.pad(image[1],((64,)*2,(64,)*2,(0,)*2,), 'constant', constant_values=0)
detector = cv2.xfeatures2d.SIFT_create(50)
for i in range(len(image)):
    keypoints, descriptors = detector.detectAndCompute(image[i], None)
    image[i] = cv2.drawKeypoints(image[i], keypoints, None, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('shift', np.hstack(image))
cv2.waitKey()
cv2.destroyAllWindows()