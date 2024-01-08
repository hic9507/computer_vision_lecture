import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# left_img = cv2.imread('./stereo/left.png')
# right_img = cv2.imread('./stereo/right.png')
#
left_img = cv2.imread('C:/Users/user/Desktop/1670491359454.JPEG')
right_img = cv2.imread('C:/Users/user/Desktop/1670491359427.JPEG')
#
# left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
# right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
#
# stereo = cv2.StereoBM_create(numDisparities=5, blockSize=15)
# disparity = stereo.compute(left_img, right_img)
# plt.imshow(disparity,'gray')
# plt.show()

stereo_bm = cv2.StereoBM_create(100)
dispmap_bm = stereo_bm.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))

stereo_sgbm = cv2.StereoSGBM_create(0, 32)
dispmap_sgbm = stereo_sgbm.compute(left_img, right_img)

plt.figure(figsize=(12, 10))

plt.subplot(221)
plt.title('left')
plt.imshow(left_img[:, :, [2, 1, 0]])

plt.subplot(222)
plt.title('right')
plt.imshow(right_img[:, :, [2, 1, 0]])

plt.subplot(223)
plt.title('BM')
plt.imshow(dispmap_bm, cmap='gray')

plt.subplot(224)
plt.title('right rectified')
plt.imshow(dispmap_sgbm, cmap='gray')
plt.show()




# dispmap_sgbm = cv2.cvtColor(dispmap_sgbm, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(dispmap_sgbm, 0, 50)
# cv2.imshow('canny', canny)
# cv2.waitKey()