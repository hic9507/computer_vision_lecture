import cv2, numpy as np

image = cv2.imread('./lena.png').astype(np.float32) / 255
print('Shape: ', image.shape)
print('Data type: ', image.dtype)
cv2.imshow('original image', image)

print("")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale')
print('Shape: ', gray.shape)
print('Data type: ', gray.dtype)
cv2.imshow('gray-scale image', gray)

print("")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print('Converted to HSV')
print('Shape: ', hsv.shape)
print('Data type: ', hsv.dtype)
cv2.imshow('hsv image', hsv)

print("")

hsv[:, :, 2] *= 2
from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shape: ', from_hsv.shape)
print('Data type: ', from_hsv.dtype)
cv2.imshow('from_hsv', from_hsv)
cv2.waitKey()
cv2.destroyAllWindows()