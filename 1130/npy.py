import numpy as np
import os

#현재 파일 이름
print(__file__)

#현재 파일 실제 경로
print(os.path.realpath(__file__))

#현재 파일 절대 경로
print(os.path.abspath(__file__))

np_load = np.load('./case1/stereo.npy')
print(np_load)