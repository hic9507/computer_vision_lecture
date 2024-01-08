import cv2, random
import argparse
import numpy as np

##### Reading image from file #####
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', default='./Lena.png', help='Image path.')
# params = parser.parse_args()
#
# img = cv2.imread(params.path)
#
# assert img is not None
#
# print('read {}'.format(params.path))
# print('shape:', img.shape)
# print('dtype:', img.dtype)
#
# img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)
# assert img is not None
# print('read {} as grayscale'.format(params.path))
# print('shape:', img.shape)
# print('dtype:', img.dtype)
##### 경계선 #####


##### OpenCV numpy.ndarray structure #####
# img = cv2.imread('lena_draw.png')
# px = img[100,100]
# print(px)
#
# blue = img[100,100,0]
# print(blue)
#
# img[100,100] = [255,255,255]
# print(img[100,100])
#
# print(img.item(10,10,2))
#
# print(img.itemset((10,10,2),100))
# print(img.item(10,10,2))
# print('img.shape: ', img.shape)
# print('img.size: ', img.size)
# print('img.dtype: ', img.dtype)
##### 경계선 #####


##### Resizing, Flipping #####
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', default='./Lena.png', help='Image path.')
# params = parser.parse_args()
# img = cv2.imread(params.path)
# print('original image shape:', img.shape)
#
# width, height = 128, 256
# resized_img = cv2.resize(img, (width, height))
# print('resized to 128x256 image shape:', resized_img.shape)
#
# w_mult, h_mult = 0.25, 0.5
# resized_img = cv2.resize(img, (0, 0), resized_img, w_mult, h_mult)
# print('0.5, 0.25 image shape:', resized_img.shape)
#
# w_mult, h_mult = 2, 4
# resized_img = cv2.resize(img, (0, 0), resized_img, w_mult, h_mult, cv2.INTER_NEAREST)
# print('2, 4 image shape:', resized_img.shape)
#
# img_flip_along_x = cv2.flip(img, 0)
# img_flip_along_x_along_y = cv2.flip(img_flip_along_x, 1)
# img_flipped_xy = cv2.flip(img, -1)
#
# assert img_flipped_xy.all() == img_flip_along_x_along_y.all()
##### 경계선 #####


##### Saving image using lossy and lossless compression
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', default='./Lena.png', help='Image path.')
# parser.add_argument('--out_png', default='./Lena_compressed.png',
#                     help='Output image path for lossless result.')
# parser.add_argument('--out_jpg', default='./Lena_compressed.jpg',
#                     help='Output image path for lossy result.')
# params = parser.parse_args()
# img = cv2.imread(params.path)
#
# cv2.imwrite(params.out_png, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#
# saved_img = cv2.imread(params.out_png)
# assert saved_img.all() == img.all()
#
# cv2.imwrite(params.out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 0])
##### 경계선 #####


##### Showing image in OpenCV window #####
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', default='./Lena.png', help='Image path.')
# parser.add_argument('--iter', default=50, help='Downsampling iterarions number.')
# params = parser.parse_args()
# orig = cv2.imread(params.path)
# orig_size = orig.shape[0:2]
#
# cv2.imshow("Original image", orig)
# cv2.waitKey(2000)
#
# resized = orig
#
# for i in range(params.iter):
#     resized = cv2.resize(cv2.resize(resized, (256, 256)), orig_size)
#     cv2.imshow("downsized&restored", resized)
#     cv2.waitKey(100)
#
# cv2.destroyWindow("downsized&restored")
#
# cv2.namedWindow("original", cv2.WINDOW_NORMAL)
# cv2.namedWindow("result")
# cv2.imshow("original", orig)
# cv2.imshow("result", resized)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
#####경계선 #####


##### Scrollbars in OpenCV window #####
# cv2.namedWindow('window')
#
# fill_val = np.array([255,255,255], np.uint8)
#
# def trackbar_callback(idx, value):
#     fill_val[idx] = value
#
# cv2.createTrackbar('R', 'window', 255, 255, lambda v: trackbar_callback(2, v))
# cv2.createTrackbar('G', 'window', 255, 255, lambda v: trackbar_callback(1, v))
# cv2.createTrackbar('B', 'window', 255, 255, lambda v: trackbar_callback(0, v))
#
# while True:
#     image = np.full((500, 500, 3), fill_val)
#     cv2.imshow('window', image)
#     key = cv2.waitKey(3)
#     if key == 27:
#         break
#
# cv2.destroyAllWindows()
##### 경계선 #####


##### Drawing 2D primitives #####
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', default='./lena.png', help='Image path.')
# params = parser.parse_args()
# image = cv2.imread(params.path)
# w, h = image.shape[1], image.shape[0]
#
# def rand_pt(mult=1.):
#     return (random.randrange(int(w * mult)),
#             random.randrange(int(h * mult)))
#
# cv2.circle(image, rand_pt(), 40, (255, 0, 0))
# cv2.circle(image, rand_pt(), 5, (255, 0, 0), cv2.FILLED)
# cv2.circle(image, rand_pt(), 40, (255, 85, 85), 2)
# cv2.circle(image, rand_pt(), 40, (255, 170, 170), 2, cv2.LINE_AA)
#
# cv2.line(image, rand_pt(), rand_pt(), (0, 255, 0))
# cv2.line(image, rand_pt(), rand_pt(), (85, 255, 85), 3)
# cv2.line(image, rand_pt(), rand_pt(), (170, 255, 170), 3, cv2.LINE_AA)
#
# cv2.arrowedLine(image, rand_pt(), rand_pt(), (0, 0, 255), 3, cv2.LINE_AA)
#
# cv2.rectangle(image, rand_pt(), rand_pt(), (255, 255, 0), 3)
#
# cv2.ellipse(image, rand_pt(), rand_pt(0.3), random.randrange(360), 0, 360, (255, 255, 255), 3)
#
# cv2.putText(image, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
#
# cv2.imshow('result', image)
# key = cv2.waitKey(0)
##### 경계선 #####


##### Handling user input from keyboard  #####
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', default='./lena.png', help='Image path.')
# params = parser.parse_args()
# image = cv2.imread(params.path)
# image_to_show = np.copy(image)
# w, h = image.shape[1], image.shape[0]
#
# def rand_pt():
#     return (random.randrange(w),
#             random.randrange(h))
#
# finish = False
# while not finish:
#     cv2.imshow('retult', image_to_show)
#     key = cv2.waitKey(0)
#     if key == ord('p'):
#         for pt in [rand_pt() for _ in range(10)]:
#             cv2.circle(image_to_show, pt, 3, (255, 0, 0), -1)
#     elif key == ord('l'):
#         cv2.line(image_to_show, rand_pt(), rand_pt(), (0, 255, 0), 3)
#     elif key == ord('r'):
#         cv2.rectangle(image_to_show, rand_pt(), rand_pt(), (0, 0, 255), 3)
#     elif key == ord('e'):
#         cv2.ellipse(image_to_show, rand_pt(), rand_pt(), random.randrange(360), 0, 360, (255, 255, 0), 3)
#     elif key == ord('t'):
#         cv2.putText(image_to_show, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
#     elif key == ord('c'):
#         image_to_show = np.copy(image)
#     elif key == 27:
#         finish = True
##### 경계선 #####


##### Handling user input from mouse #####
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', default='./lena.png', help='Image path.')
# params = parser.parse_args()
# image = cv2.imread(params.path)
# image_to_show = np.copy(image)
# w, h = image.shape[1], image.shape[0]
#
# mouse_presssed = False
# s_x = s_y = e_x = e_y = -1
#
# def mouse_callback(event, x, y, flags, param):
#     global image_to_show, s_x, s_y, e_x, e_y, mouse_presssed
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         mouse_presssed = True
#         s_x, s_y = x, y
#         image_to_show = np.copy(image)
#
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if mouse_presssed:
#             image_to_show = np.copy(image)
#             cv2.rectangle(image_to_show, (s_x, s_y),
#                           (x, y), (0, 255, 0), 1)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         mouse_presssed = False
#         e_x, e_y = x, y
#
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', mouse_callback)
#
# while True:
#     cv2.imshow('image', image_to_show)
#     k = cv2.waitKey(1)
#
#     if k == ord('c'):
#         if s_y > e_y:
#             s_y, e_y = e_y, s_y
#         if s_x > e_x:
#             s_x, e_x = e_x, s_x
#
#         if e_y - s_y > 1 and e_x - s_x > 0:
#             image = image[s_y:e_y, s_x:e_x]
#             image_to_show = np.copy(image)
#     elif k == 27:
#         break
# cv2.destroyAllWindows()
##### 경계선 #####


##### Playing frame stream from video #####
# capture = cv2.VideoCapture('./0921-1.mp4')
#
# while True:
#     has_frame, frame = capture.read()
#     if not has_frame:
#         print('Reached end of video')
#         break
#
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(500)
#     if key == 27:
#         print('Pressed Esc')
#         break
# cv2.destroyAllWindows()
##### 경계선 #####


import cv2, random
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./lena.png', help='Image path.')
params = parser.parse_args()
image = cv2.imread(params.path)
image_to_show = np.copy(image)
w, h = image.shape[1], image.shape[0]

image = cv2.imread('./lena.png')
# cv2.imshow('image', image)
# cv2.waitKey(0)
mouse_presssed = False
s_x = s_y = e_x = e_y = -1

def mouse_callback1(event, x, y, flags, param):

    global image_to_show, s_x, s_y, e_x, e_y, mouse_presssed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_presssed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image_to_show)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_presssed = False
        cv2.rectangle(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1)
        cv2.imshow('image_to_showz', image)

def mouse_callback2(event, x, y, flags, param):

    global image_to_show, s_x, s_y, e_x, e_y, mouse_presssed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_presssed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image_to_show)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_presssed = False
        cv2.line(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 3)
        # cv2.imshow('image_to_show', image)

def mouse_callback3(event, x, y, flags, param):

    global image_to_show, s_x, s_y, e_x, e_y, mouse_presssed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_presssed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image_to_show)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_presssed = False
        cv2.arrowedLine(image, (s_x, s_y), (x, y), (0, 0, 255), 3, cv2.LINE_AA)
        # cv2.imshow('image_to_show', image)



finish = False
while not finish:
    cv2.imshow('aa', image_to_show)
    k = cv2.waitKey(0)

    if k == ord('r'):
        cv2.setMouseCallback('image_to_show1', mouse_callback1)

    elif k == ord('l'):
        cv2.setMouseCallback('image_to_show2', mouse_callback2)

    elif k == ord('a'):
        cv2.setMouseCallback('image_to_show3', mouse_callback3)

    elif k == ord('w'):
        cv2.imwrite('./lena_draw.png', image_to_show)

    elif k == ord('c'):
        image_to_show = np.copy(image)

    elif k == 27:
        cv2.destroyAllWindows()
