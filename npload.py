import numpy as np
import os
import sys

dir = 'C:/Users/user/Desktop/env/RTFM/tpr.npy'
np.set_printoptions(threshold=sys.maxsize)

a = np.array(np.load(dir)[:])

print(a)
print(len(a))