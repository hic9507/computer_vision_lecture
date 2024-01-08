import cv2
import numpy as np
import matplotlib.pyplot as plt

# image = cv2.imread('./BnW.png', 0)
image = cv2.imread('./Lena.png', 0)
otsu_thr, otsu_mask = cv2.threshold(image, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

image = otsu_mask

# dst, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]

image_external = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2. drawContours(image_external, contours, i, 255, -1)


image_internal = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.axis('off')
plt.title('thresholding')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('externel')
plt.imshow(image_external, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('internal')
plt.imshow(image_internal, cmap='gray')
plt.tight_layout()
plt.show()