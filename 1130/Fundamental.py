import cv2
import numpy as np
import matplotlib.pyplot as plt
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

data = np.load('C:/Users/user/Desktop/computer_vision/1130/case1/stereo.npy').item()
Kl, Kr, Dl, Dr, left_pts, right_pts, E_from_stereo, F_from_stereo = \
    data['Kl'], data['Kr'], data['Dl'], data['Dr'], data['left_pts'], data['right_pts'], data['E'], data['F']

left_pts = np.vstack(left_pts)
right_pts = np.vstack(right_pts)

left_pts = cv2.undistortPoints(left_pts, Kl, Dl, P=Kl)
right_pts = cv2.undistortPoints(right_pts, Kr, Dr, P=Kr)

F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)

E = Kr.T @ F @ Kl

print('Fundamental matrix: ')
print(F)
print('Essential matrix: ')
print(E)

print('-'*100)
print(F[0])
print(F[0][0])

A = [F[0][0], F[1][0], F[2][0], F[0][1], F[1][1], F[2][1]]
B = [E[0][0], E[1][0], E[2][0], E[0][1], E[1][1], E[2][1]]

# A = F[1][2]
# B = F[2][1]

plt.plot(A, B)
plt.show()