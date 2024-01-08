import cv2
import matplotlib.pyplot as plt
import numpy as np

########################################################################### 1번 시작 ########################################################################
# Canny Edge
img0 = cv2.imread('./stitching/boat1.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('./stitching/budapest1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_GRAYSCALE)

# img0 = cv2.resize(img0, (256, 256))
# img1 = cv2.resize(img1, (256, 256))
# img2 = cv2.resize(img2, (256, 256))
# img3 = cv2.resize(img3, (256, 256))

edge0 = cv2.Canny(img0, 200, 100)
edge1 = cv2.Canny(img1, 200, 100)
edge2 = cv2.Canny(img2, 200, 100)
edge3 = cv2.Canny(img3, 200, 100)

cv2.imshow('edge0', edge0)
cv2.imshow('edge1', edge1)
cv2.imshow('edge2', edge2)
cv2.imshow('edge3', edge3)
cv2.waitKey()
cv2.destroyAllWindows()
# cv2.destroyAllWindows()

# Harris Corner
img0 = cv2.imread('./stitching/boat1.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('./stitching/budapest1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_COLOR)
img3 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_COLOR)

corners0 = cv2.cornerHarris(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
corners1 = cv2.cornerHarris(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
corners2 = cv2.cornerHarris(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
corners3 = cv2.cornerHarris(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)

corners0 = cv2.dilate(corners0, None)
corners1 = cv2.dilate(corners1, None)
corners2 = cv2.dilate(corners2, None)
corners3 = cv2.dilate(corners3, None)

show_img0 = np.copy(img0)
show_img1 = np.copy(img1)
show_img2 = np.copy(img2)
show_img3 = np.copy(img3)

show_img0[corners0 > 0.1 * corners0.max()] = [0, 0, 255]
show_img1[corners1 > 0.1 * corners1.max()] = [0, 0, 255]
show_img2[corners2 > 0.1 * corners2.max()] = [0, 0, 255]
show_img3[corners3 > 0.1 * corners3.max()] = [0, 0, 255]

corners0 = cv2.normalize(corners0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
corners1 = cv2.normalize(corners1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
corners2 = cv2.normalize(corners2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
corners3 = cv2.normalize(corners3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

show_img0 = np.hstack((show_img0, cv2.cvtColor(corners0, cv2.COLOR_GRAY2BGR)))
show_img1 = np.hstack((show_img1, cv2.cvtColor(corners1, cv2.COLOR_GRAY2BGR)))
show_img2 = np.hstack((show_img2, cv2.cvtColor(corners2, cv2.COLOR_GRAY2BGR)))
show_img3 = np.hstack((show_img3, cv2.cvtColor(corners3, cv2.COLOR_GRAY2BGR)))

cv2.imshow('Harris corner detector_boat1', show_img0)
cv2.imshow('Harris corner detector_budapest1', show_img1)
cv2.imshow('Harris corner detector_newspaper1', show_img2)
cv2.imshow('Harris corner detector_s1', show_img3)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

####################################################################### 1번 끝 #####################################################################