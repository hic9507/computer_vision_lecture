import cv2
import numpy as np

camera_matrix = np.load('C:/Users/user/Desktop/computer_vision/1123/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('C:/Users/user/Desktop/computer_vision/1123/pinhole_calib/dist_coefs.npy')
img = cv2.imread('C:/Users/user/Desktop/computer_vision/1123/pinhole_calib/img_00.png')

pattern_size = (10, 7)
res, corners = cv2.findChessboardCorners(img, pattern_size)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                        corners, (10, 10), (-1, -1), criteria)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)