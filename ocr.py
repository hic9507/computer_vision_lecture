import os
import numpy as np
import sys

# sys.stdout = open('./labeling.txt', 'w', encoding='cp949')
ocr_f = open('./ocr_label.txt', 'a', encoding='utf-8')

f_train = open('D:/ocr_share/data_v2/train_gt.txt', 'r', encoding='utf-8')

lines = f_train.readlines()  # 2807123


for line in lines:
    line = line.split('\t')  # ['training/5350177-1995-0001-0001_1.jpg', '도시계획과\n']


    character = line[1]  # '도시계획과\n'


    # line[0] # training/5350177-1995-0001-0001_1.jpg


    character_ = line[0].split('/')  # ['training', '5350177-2001-0001-0010_78.jpg']
    character__ = line[0].split('/')[1]  # 5350177-2001-0001-0010_78.jpg


    for i in os.listdir('D:/ocr_share/data_v2/training/'):
        path = 'D:/ocr_share/data_v2/training/' + i    # 폴더 경로 + 이미지 파일 이름
        path_ = path.split('/') # ['D:', 'ocr_share', 'data_v2', 'training', '0101351A34FC82B65D51CB1BA7AD7616_013.jpg']
        
        name = path_[4]

        if name == character__:
            print('D:/ocr_share/data_v2/training/' + name + '\t' + character[:-1], file=ocr_f)




        # #     # print('D:/ocr_share/data_v2/training/' + character__[path_[4]] + '\t' +character[path_[4]] + '\n')

    # if character in line:
    #     ocr_file = line[0] + '\t' + character
    #     # print(ocr_file) #, file=ocr_f
    # else:
    #     print('x')

# sys.stdout.close()
# ocr_f.write(ocr_file)