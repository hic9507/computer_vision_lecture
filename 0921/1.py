import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./lena.png')
print(img.shape)
img_to_show = np.copy(img)

r = img[:, :, 2]
g = img[:, :, 1]
b = img[:, :, 0]

print("rgb 중 하나를 키보드로 입력하시오.")

while True:
    cv2.imshow('img', img_to_show)
    k = cv2.waitKey(1)

    if k == ord('r'):
        img = cv2.imread('./lena.png')
        hist, bins = np.histogram(r, 256, [0, 255])
        plt.fill(hist)
        plt.xlabel('before equalization(pixel value of r)')
        plt.show()
        r_eq = cv2.equalizeHist(r)
        hist, bins = np.histogram(r_eq, 256, [0, 255])
        plt.fill_between(range(256), hist, 0)
        plt.xlabel('after equalization(pixel value of r)')
        plt.show()
        img[:, :, 2] = r_eq
        cv2.imshow('r', img)

    elif k == ord('g'):
        img = cv2.imread('./lena.png')
        hist, bins = np.histogram(g, 256, [0, 255])
        plt.fill(hist)
        plt.xlabel('before equalization(pixel value of g)')
        plt.show()
        g_eq = cv2.equalizeHist(g)
        hist, bins = np.histogram(g_eq, 256, [0, 255])
        plt.fill_between(range(256), hist, 0)
        plt.xlabel('after equalization(pixel value of g)')
        plt.show()
        img[:, :, 1] = g_eq
        cv2.imshow('g', img)

    elif k == ord('b'):
        img = cv2.imread('./lena.png')
        hist, bins = np.histogram(b, 256, [0, 255])
        plt.fill(hist)
        plt.xlabel('before equalization(pixel value of b)')
        plt.show()
        b_eq = cv2.equalizeHist(b)
        hist, bins = np.histogram(b_eq, 256, [0, 255])
        plt.fill_between(range(256), hist, 0)
        plt.xlabel('after equalization(pixel value of b)')
        plt.show()
        img[:, :, 0] = b_eq
        cv2.imshow('b', img)

    elif k == 27:
        break

# image = cv2.imread('./Lena.png', cv2.IMREAD_COLOR)
# rgb = input("r,g,b 중 하나를 입력하세요 : ")
#
# if rgb == 'r':
#     hist, bins = np.histogram(image[:, :, 2], 256, [0, 255])
#     plt.fill(hist)
#     plt.xlabel('pixel value')
#     plt.show()
#     r = cv2.equalizeHist(image[:, :, 2])
#     image[:, :, 2] = r
#     cv2.imshow('r', image)
#     cv2.waitKey()
#
# elif rgb == 'g':
#     hist, bins = np.histogram(image[:, :, 1], 256, [0, 255])
#     plt.fill(hist)
#     plt.xlabel('pixel value')
#     plt.show()
#     g = cv2.equalizeHist(image[:, :, 1])
#     image[:, :, 1] = g
#     cv2.imshow('g', image)
#     cv2.waitKey()
#
# elif rgb == 'b':
#     hist, bins = np.histogram(image[:, :, 0], 256, [0, 255])
#     plt.fill(hist)
#     plt.xlabel('pixel value')
#     plt.show()
#     b = cv2.equalizeHist(image[:, :, 0])
#     hist, bins = np.histogram(b, 256, [0, 255])
#     plt.fill_between(range(256), hist, 0)
#     plt.xlabel('pixel value of r')
#     plt.show()
#     image[:, :, 0] = b
#     cv2.imshow('b', image)
#     cv2.waitKey()

# cv2.imshow('aa',image)
# cv2.waitKey()
# cv2.destroyAllWindows()

