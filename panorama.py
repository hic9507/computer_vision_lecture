import cv2
import numpy as np
import os
from PIL import Image

images = []
img_dir = 'C:/Users/user/Desktop/computer_vision/img/'
for i in os.listdir(img_dir):
    path = img_dir + i
    image = np.array(Image.open(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(gray)
    

# images = [image0, image1]
stitcher = cv2.createStitcher()
ret, pano = stitcher.stitch(images)

if ret == cv2.STITCHER_OK:
    pano = cv2.resize(pano, dsize=(0, 0), fx=0.2, fy=0.2)
    pano = cv2.resize(pano, (512, 512))
    cv2.imshow('panorama', pano)
    cv2.waitKey()

    cv2.destroyAllWindows()
else:
    print('Error during stiching')