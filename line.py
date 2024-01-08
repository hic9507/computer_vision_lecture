import os
import numpy as np
import sys
# sys.stdout = open('./labeling.txt', 'w', encoding='cp949')
ocr_f = open('./labeling1.txt', 'a', encoding='cp949')

f_train = open('D:/ocr_share/data_v2/train_gt.txt', 'r', encoding='cp949')

lines = f_train.readlines()

for line in lines:
    line = line.split('\t')

    character = line[1]


    character_ = line[0].split('/') # ['training', '5350177-2001-0001-0010_78.jpg']
    character__ = line[0].split('/')[1] # '5350177-2001-0001-0010_78.jpg'
    
    
    for i in os.listdir('D:/ocr_share/data_v2/training/'):
        path = 'D:/ocr_share/data_v2/training/' + i
        
        path_ = path.split('/')
        # print(path_[4]) # 05010012040_004.jpg

        if path_[4] == character__:
            # print('D:/ocr_share/data_v2/training/' + character__[path_[4]] + '\t' +character[path_[4]] + '\n')
            print('D:/ocr_share/data_v2/training/' + path_[4] + '\t' + character[:-1])


    # if character in line:
    #     ocr_file = line[0] + '\t' + character
    #     # print(ocr_file) #, file=ocr_f
    # else:
    #     print('x')




# sys.stdout.close()
# ocr_f.write(ocr_file)

    # 
    #     path = 'D:/ocr_share/data_v2/training/' + i

    