import argparse
from xml.dom import HierarchyRequestErr
import cv2
import math
from keras.preprocessing import image
import numpy as np
import matplotlib as plt
# from skimage.measure import compare_ssim
import matplotlib as plt
import random

parser = argparse.ArgumentParser()

parser.add_argument('--path', default='lena.jfif', help='Image path')
params = parser.parse_args()

img = cv2.imread(params.path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

otsu_thr, otsu_mask = cv2.threshold(img, -1,1,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('Estimated threshold (Otsu):', otsu_thr)


otsu_mask = otsu_mask*255

cv2.imshow('otsu_mask', otsu_mask)
cv2.waitKey()



contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

image_external = np.zeros(otsu_mask.shape, otsu_mask.dtype)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)

image_internal = np.zeros(otsu_mask.shape, otsu_mask.dtype)

for i in range(len(contours)):
    if hierarchy[0][i][3]!=-1:
        cv2.drawContours(image_internal, contours,i, 255, -1)


# print(image_external)

cv2.imshow('image_external', image_external)
cv2.imshow('image_internal', image_internal)
cv2.waitKey()

connectivity=8


img = cv2.imread(params.path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)

num_labels, labelmap, stats, centers = output

colored = np.full((otsu_mask.shape[0], otsu_mask.shape[1], 3), 0, np.uint8)

key = cv2.waitKey(0)



r = []
for i in range(1, num_labels):
    if stats[i][4]>200:
        r.append(i)




while(len(r)):
    cv2.imshow('orign', img)
    if key ==ord(' '):

        x = r.pop(random.randrange(0,len(r)))

        colored[labelmap == x]=(0,255*x/num_labels, 255*(num_labels-x)/num_labels)

        cv2.circle(colored,
        (int(centers[x][0]),int(centers[x][1])),5,(255,0,0), cv2.FILLED)

        img = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Connected components', np.hstack((img, colored)))
        cv2.waitKey()



distmap = cv2.distanceTransform(otsu_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


cv2.imshow('distmap', distmap)
cv2.waitKey()




#
# img = np.full((512,512,3), 255, np.uint8)
#
# axes = (int(256*random.uniform(0,1)), int(256*random.uniform(0,1)))
# angle = int(180*random.uniform(0,1))
#
#
# center = (256,256)
#
# pts = cv2.ellipse2Poly()
#

# finish = False

# while not finish:


#     key = cv2.waitKey(0)

#     if key ==ord(' '):
#         output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)

#         num_labels, labelmap, stats, centers = output

#         colored = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8)

#         for i in range(1, num_labels):
#             if stats[i][4]>200:
#                 colored[labelmap == i]=(0,255*i/num_labels, 255*(num_labels-i)/num_labels)
#                 cv2.circle(colored,
#                 (int(centers[i][0]),int(centers[i][1])),5,(255,0,0), cv2.FILLED)

        # img = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('Connected components', np.hstack((img, colored)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
