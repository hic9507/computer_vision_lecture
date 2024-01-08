import os
import numpy as np
import sys

# ocr_f = open('./value.txt', 'a', encoding='cp949')
ocr_f = open('./value2.txt', 'a', encoding='cp949')
ocr_f1 = open('./value3.txt', 'a', encoding='cp949')

# f_train = open('./ocr_label.txt', 'r', encoding='cp949')
f_train = open('./value.txt', 'r', encoding='cp949')


# lines = f_train.readlines()
lines = f_train.read()
result = lines.replace('\n', '')
print(result, end='', file=ocr_f1)#, file=ocr_f1

# for line in lines:
#     line = line.strip(' ')
#     result = line.replace(" ", "")
#     result1 = result.replace('\n', "")

#     # character = line[1][:-1]
#     # character1 = character.strip('\n')
#     # print(character1)
#     # print(result1, end='', file=ocr_f)
#     print(result1)


    # character_ = line[0].split('/') # ['training', '5350177-2001-0001-0010_78.jpg']
    # character__ = line[0].split('/')[1] # '5350177-2001-0001-0010_78.jpg'
   