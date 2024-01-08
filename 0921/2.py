# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = cv2.imread('./Lena.png', 0).astype(np.float32) / 255
#
# fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
# fft_shift = np.fft.fftshift(fft, axes=[0, 1])
# sz = 25
# mask = np.zeros(fft.shape, np.uint8)
# mask[image.shape[0]//2-sz:image.shape[0]//2+sz,
#     image.shape[1]//2-sz:image.shape[1]//2+sz, :] = 1
# fft_shift *= mask
# fft = np.fft.ifftshift(fft_shift, axes=[0, 1])
#
# filterd = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
# mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))
#
# plt.figure()
# plt.subplot(131)
# plt.axis('off')
# plt.title('original')
# plt.imshow(image, cmap='gray')
# plt.subplot(132)
# plt.axis('off')
# plt.title('no high frequencies')
# plt.imshow(filterd, cmap='gray')
# plt.subplot(133)
# plt.axis('off')
# plt.title('mask')
# plt.imshow(mask_new*255, cmap='gray')
# plt.tight_layout(True)
# plt.show()

import cv2, numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./lena.png').astype(np.float32) / 255

noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)
# plt.imshow(noised[:, :, [2, 1, 0]])
# plt.show()

# gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
# plt.imshow(gauss_blur[:, :, [2, 1, 0]])
# plt.show()

# median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
# plt.imshow(median_blur[:, :, [2, 1, 0]])
# plt.show()
#
# bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
# plt.imshow(bilat[:, :, [2, 1, 0]])
# plt.show()

diameter = int(input('diamter: '))
SigmaColor = float(input('SigmaColor: '))
SigmaSpce = int(input('SigmaSpace: '))

bilat = cv2.bilateralFilter(noised, diameter, SigmaColor, SigmaSpce)
bilat_img = bilat
cv2.imshow('bilat_img', bilat_img) # remove noise
cv2.imshow('nosed_img', noised)
cv2.waitKey()