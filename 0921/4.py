import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./Lena.png', 0)
_, binary = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

n_erosion = int(input('num of erosion: '))
n_dilation = int(input('num of dilation: '))
n_opening = int(input('num of opening: '))
n_closing = int(input('num of closing: '))

eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3,3), iterations=n_erosion)
dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3,3), iterations=n_dilation)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=n_opening)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=n_closing)

grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

plt.figure(figsize=(10,10))
# plt.subplot(231)
# plt.axis('off')
# plt.title('binary')
# plt.imshow(binary, cmap='gray')
plt.subplot(221)
plt.axis('off')
plt.title('erode ' + str(n_erosion) + ' times')
plt.imshow(eroded, cmap='gray')
plt.subplot(222)
plt.axis('off')
plt.title('dilated ' + str(n_dilation) + ' times')
plt.imshow(dilated, cmap='gray')
plt.subplot(223)
plt.axis('off')
plt.title('open ' + str(n_opening) + ' times')
plt.imshow(opened, cmap='gray')
plt.subplot(224)
plt.axis('off')
plt.title('closed ' + str(n_closing) + ' times')
plt.imshow(closed, cmap='gray')
# plt.subplot(236)
# plt.axis('off')
# plt.title('gradient')
# plt.imshow(grad, cmap='gray')
plt.tight_layout()
plt.show()