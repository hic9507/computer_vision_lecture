import cv2
import numpy as np
import matplotlib.pyplot as plt

######################## DFT.py #####################
image = cv2.imread('./Lena.png', 0).astype(np.float32) / 255

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)

shifted = np.fft.fftshift(fft, axes=[0, 1]) # 주파수 재배열 함수로, 주파수가 0인 부분을 정중앙에, 주파수가 커질수록 가장자리에 위치시킴.
magnitude = cv2.magnitude(shifted[:,:,0], shifted[:, :, 1]) # 벡터 크기 계산
magnitude = np.log(magnitude)

plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout(True)
plt.show()

################################# frequency.py ###################
image = cv2.imread('./Lena.png', 0).astype(np.float32) / 255

rad = int(input('radius: '))

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])
sz = 25

print(fft.shape)
# mask[image.shape[0]//2-sz:image.shape[0]//2+sz, image.shape[1]//2-sz:image.shape[1]//2+sz, :] = 1
i = ' '
i = input('Low pass or High pass: ')

if i == 'Low pass':
    mask = np.zeros(fft.shape, np.uint8)
    cv2.circle(mask, (int(image.shape[1]/2), int(image.shape[0]/2)), int(rad), (1, 1), -1)
elif i == 'High pass':
    mask = np.ones(fft.shape, np.uint8)
    cv2.circle(mask, (int(image.shape[1]/2), int(image.shape[0]/2)), int(rad), (0, 0), -1)
fft_shift *= mask
fft = np.fft.ifftshift(fft_shift, axes=[0, 1])
# print(fft_shift)
filterd = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))

plt.figure()
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('no high frequencies')
plt.imshow(filterd, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('mask')
plt.imshow(mask_new*255, cmap='gray')
plt.tight_layout(True)
plt.show()