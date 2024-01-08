import cv2
import numpy as np
import os

pattern_size = (8, 6)
samples = []

file_list = os.listdir('./pinhole_calib')
img_file_list = [file for file in file_list if file.startswith('img')]