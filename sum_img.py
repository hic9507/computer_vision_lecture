import numpy as np
from PIL import Image
import cv2
import numpy as np
import os

images = []
# img_dir = 'C:/Users/user/Desktop/computer_vision/img/'
img_dir = 'C:/Users/user/Desktop/env/RTFM/SH_Train_ten_crop_i3d/'
image = np.load(os.path.join(img_dir+'01_002_i3d.npy'))

print(image)
print(image.shape)

# for i in os.listdir(img_dir):
#     path = img_dir + i
#     image = np.load(path)
#     print(image.shape)
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     images.append(image)
#     # print(len(images))
#     # print(images.shape)

#     # print(len(images[0]))
#     # print(images[0][0])

#     # break
#     # sum = 0
#     # sum += images[i]
#     a = np.concatenate(images[0])
# a = a.T
# pil_image=Image.fromarray(a[:], 'L')
# pil_image.show()
# pil_image.save('./transpose/{}.jpg'.format(i))
    # print(a)