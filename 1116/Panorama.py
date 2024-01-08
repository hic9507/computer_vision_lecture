import cv2
import numpy as np

images = []
# images.append(cv2.imread('./panorama/0.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('./panorama/1.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('C:/Users/user/Desktop/computer_vision/final/image/bs_1.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('C:/Users/user/Desktop/computer_vision/final/image/1.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('C:/Users/user/Desktop/computer_vision/final/image/1.jpg', cv2.IMREAD_COLOR))
images1 = cv2.imread('C:/Users/user/Desktop/computer_vision/final/image/bs_1.jpg', cv2.IMREAD_COLOR)
images2 = cv2.imread('C:/Users/user/Desktop/computer_vision/final/image/bs_2.jpg', cv2.IMREAD_COLOR)
images3 = cv2.imread('C:/Users/user/Desktop/computer_vision/final/image/bs_3.jpg', cv2.IMREAD_COLOR)

image0 = cv2.imread('./000.jpg', cv2.IMREAD_COLOR)
image1 = cv2.imread('./001.jpg', cv2.IMREAD_COLOR)
# image2 = cv2.imread('./id2.jpg', cv2.IMREAD_COLOR)
# image3 = cv2.imread('./id3.jpg', cv2.IMREAD_COLOR)
print(image0.shape)
print(image1.shape)
# print(image2.shape)
# print(image3.shape)


# images = [image0, image1, image2, image3]
images = [image0, image1]
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