import cv2
import numpy as np
import matplotlib.pyplot as plt

# (1)
img =cv2.imread('./Lena.png', 0)

otsu_thr, otsu_mask = cv2.threshold(img, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('Estimated threshold (Otsu):', otsu_thr)
img = otsu_mask

# (2)
contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]

image_external = np.zeros(img.shape, img.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2. drawContours(image_external, contours, i, 255, -1)


image_internal = np.zeros(img.shape, img.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(img, cmap='gray')
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

# (3)
connectivity = 8
# num_lables, labelmap = cv2.connectedComponents(img, connectivity, cv2.CV_32S)

output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)

num_lables, labelmap, stats, centers = output

colored = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8)

for l in range(1, num_lables):
    if stats[l][4] > 200:
        while True:
            key = cv2.waitKey()
            if cv2.waitKey() == ord('q'):
                colored[labelmap == l] = (0, 255*l/num_lables, 255*(num_lables-l)/num_lables)
                cv2.circle(colored,
                           (int(centers[l][0]), int(centers[l][1])), 5, (255, 0, 0), cv2.FILLED)
            elif key == 27:
                break

img = cv2.cvtColor(otsu_mask*255, cv2.COLOR_GRAY2BGR)

cv2.imshow('Connected components', np.hstack((img, colored)))
cv2.waitKey()
cv2.destroyAllWindows()