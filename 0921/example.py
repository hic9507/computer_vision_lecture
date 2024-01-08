import cv2, numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lena.png')
cv2.imshow('lena_color', image)

gray = cv2.imread('lena.png', 0)
cv2.imshow('lena_gray', gray)
cv2.waitKey()

# gray = cv2.imread('./lena.png', 0).astype(np.float32) / 255
#
# gamma = 0.5
# corrected_image = np.power(gray, gamma)
#
# cv2.imshow('image', image)
# cv2.imshow('corrected_image', corrected_image)
# cv2.waitKey()
# gray = cv2.imread('./lena.png', 0).astype(np.float32) / 255
gray_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(gray_eq, 256, [0, 255])

gray_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(gray_eq, 256, [0, 255])
cv2.imshow('equalized grey', gray_eq)

a = cv2.imread('./lena.png', 0).astype(np.float32) / 255
gamma = 0.5
corrected_image = np.power(a, gamma)
cv2.imshow('gamma_corrected_image', corrected_image)
cv2.waitKey()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h = hsv[:, :, [0]]
s = hsv[:,[0] ,:]
v = hsv[[0], :, :]

h,s,v = cv2.split(hsv)

cv2.imshow('hsv', hsv)

cv2.imshow('h_ori', h)
cv2.imshow('s_ori', s)
cv2.imshow('v_ori', v)
cv2.waitKey()

# cv2.imshow('h', h)
# cv2.imshow('s', s)
# cv2.imshow('v', v)
# cv2.waitKey()
# cv2.destroyAllWindows()
# noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
# noised = noised.clip(0, 1)

h = cv2.medianBlur(h, 7)
s = cv2.GaussianBlur((h*255), (7, 7), 0)
v = cv2.bilateralFilter(v, -1, 0.3, 10)

cv2.imshow('h: Medianblur', h)
cv2.imshow('s: GaussianBlur', s)
cv2.imshow('v: bilateralFilter', v)
cv2.waitKey()