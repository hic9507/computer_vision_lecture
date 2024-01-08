import numpy as np
from PIL import Image

image = np.load('C:/Users/user/Desktop/env/RTFM/SH_Train_ten_crop_i3d/01_002_i3d.npy')
print(type(image))
# np_array = np.array(image)
print(type(image))

print(image.shape)
print(image)

for i in range(2048):
    pil_image=Image.fromarray(image[:,:,i],  'L')
    # pil_image.show()
    # pil_image.save('./img/{}.jpg'.format(i))
    sum1 = np.concatenate((pil_image))
    print(sum1)

#('uint8'), 'L'

