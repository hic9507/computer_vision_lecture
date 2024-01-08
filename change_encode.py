# -*- coding: utf-8 -*-
import os
# import 
ocr = open('./change_encode.txt', 'a', encoding='utf-8')
a = open('C:/Users/user/Desktop/test/label_train.txt', 'r' , encoding='utf-8')
lines = a.readlines()

for line in lines:
    line = line.split('\t')
    character = line[1]
    # print(line[0])
    print(line[0] + '\t' + character[:-1], file=ocr)
# print(f, file=ocr)