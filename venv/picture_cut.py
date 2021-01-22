# import tensorflow as tf
import cv2
import random
#准备把图片切割成多个若干128*128的小图片
n = 1
for i in range(1,9):
    data = cv2.imread('C:\\Users\\14999\\Desktop\\dataset\Data{}.png'.format(i))
    label = cv2.imread('C:\\Users\\14999\\Desktop\\label\label{}.png'.format(i))
    print(data.shape)
    for j in range(0,100):
        y = random.randint(0,471)
        x = random.randint(0,371)
        data1 = data[x:x+128,y:y+128]
        label1 = label[x:x+128,y:y+128]
        cv2.imwrite('C:\\Users\\14999\\Desktop\\data_cut\\data{}.png'.format(n),data1)
        cv2.imwrite('C:\\Users\\14999\\Desktop\\label_cut\\label{}.png'.format(n),label1)
        n+=1
