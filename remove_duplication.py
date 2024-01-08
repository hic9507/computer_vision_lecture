import os

f_train = open('C:/Users/user/Desktop/computer_vision/ocr_label.txt', 'r', encoding='cp949')
ocr_f = open('./ocr_label_test.txt', 'a', encoding='cp949')

lines = f_train.readlines()  # 2807123

for i in lines:
    # print(i)
    for line in lines:
        if line == i:
            print(line, file=ocr_f)