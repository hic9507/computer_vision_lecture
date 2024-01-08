import cv2
import numpy as np
import os
import glob

PATTERN_SIZE = (9, 6)
left_imgs = list(sorted(glob.glob('./data')))