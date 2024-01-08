import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

image = cv2.imread('./Lena.png',0)

KSIZE = 11
ALPHA = 2
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA
# print(kernel.shape, kernel.dtype, kernel.sum())

filtered = cv2.filter2D(image, -1, kernel)

# plt.figure(figsize=(8,4))
# plt.subplot(121)
# plt.axis('off')
# plt.title('image')
# plt.imshow(image[:, :, [2, 1, 0]])
# plt.subplot(122)
# plt.axis('off')
# plt.title('filtered')
# plt.imshow(filtered[:, :, [2, 1, 0]])
# plt.tight_layout(True)
# plt.show()
### (1)
cv2.imshow('before', image)
cv2.imshow('after', filtered)

# image = cv2.imread('./Lena.png', 0)
### (2)
dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
cv2.imshow('dx', dx)
cv2.imshow('dy', dy)
cv2.waitKey()


### (3)
kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel * kernel).sum())

filtered = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('kernel')
plt.imshow(kernel, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered, cmap='gray')
plt.tight_layout()
plt.show()

### (4)
# def call_back(pos) :
#     pass
#
# image = cv2.imread("./Lena.png", 0)
# cv2.imshow('image', image)
# cv2.createTrackbar('sobel', 'image', 0, 255, call_back)
# cv2.createTrackbar('gabor', 'image', 0, 255, call_back)
# cv2.setTrackbarPos('threshold', 'image', 127)
#
# while True :
#     low = cv2.getTrackbarPos('threshold', 'image')
#     ret, image_binary = cv2.threshold(image, low, 255, cv2.THRESH_BINARY)
#     cv2.imshow('image', image_binary)
#     if cv2.waitKey(1) == 27 :
#         break
### (4)
def onChange(pos):
    pass

src = cv2.imread("./Lena.png", 0)

cv2.namedWindow("image")

cv2.createTrackbar("threshold", "image", 0, 255, onChange)
cv2.createTrackbar("maxValue", "image", 0, 255, lambda x : x)

cv2.setTrackbarPos("threshold", "image", 127)
cv2.setTrackbarPos("maxValue", "image", 255)

while cv2.waitKey(1) !=27:

    thresh = cv2.getTrackbarPos("threshold", "image")
    maxval = cv2.getTrackbarPos("maxValue", "image")

    _, binary = cv2.threshold(src, thresh, maxval, cv2.ADAPTIVE_THRESH_MEAN_C)

    cv2.imshow("image", binary)

cv2.destroyAllWindows()