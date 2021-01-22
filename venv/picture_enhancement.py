import cv2
import tensorflow as tf
import numpy as np
import PIL,os
import skimage.io as io
###################################################################################
#旋转或镜像
# for i in range(1,801):
#     data = cv2.imread(r'C:\Users\14999\Desktop\data_cut\data{}.png'.format(i))
#     label = cv2.imread(r'C:\Users\14999\Desktop\label_cut\label{}.png'.format(i))
#     for j in range(0,4):
#         if j==0:
#             data1 = cv2.transpose(data)
#             label1 = cv2.transpose(label)
#         elif j==1:
#             data1 = cv2.flip(data,-1)
#             label1 = cv2.flip(label,-1)
#         elif j==2:
#             data1 = cv2.flip(data,0)
#             label1 = cv2.flip(label,0)
#         elif j==3:
#             data1 = cv2.flip(data,1)
#             label1 = cv2.flip(label,1)
#         print(data1)
#         print(label1)
#         cv2.imwrite(r'C:\Users\14999\Desktop\data\data{}_{}.png'.format(i,j),data1)
#         cv2.imwrite(r'C:\Users\14999\Desktop\label\label{}_{}.png'.format(i,j),label1)
###########################################################################################
#模糊
# for i in range(1,801):
#     data = cv2.imread(r'C:\Users\14999\Desktop\data_cut\data{}.png'.format(i))
#     label = cv2.imread(r'C:\Users\14999\Desktop\label_cut\label{}.png'.format(i))
#     for j in range(4,6):
#         if j==4:
#             #均值模糊
#             data1 = cv2.blur(data,(5,5))
#         else:
#             #中值噪声
#             data1 = cv2.medianBlur(data,5)
#         cv2.imwrite(r'C:\Users\14999\Desktop\data\data{}_{}.png'.format(i,j),data1)
#         cv2.imwrite(r'C:\Users\14999\Desktop\label\label{}_{}.png'.format(i,j),label)
##########################################################################################
#亮度
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.1,1.5])
for i in range(1,801):
    data = cv2.imread(r'C:\Users\14999\Desktop\data_cut\data{}.png'.format(i))
    label = cv2.imread(r'C:\Users\14999\Desktop\label_cut\label{}.png'.format(i))
    j = 6
    # data = tf.keras.preprocessing.image.img_to_array(data)
    X = []
    X.append(data)
    X = np.array(X)
    for batch in train_datagen.flow(X,batch_size=5,save_to_dir=r'C:\Users\14999\Desktop\1',save_format='png',save_prefix='data{}_{}'.format(i,j)):
        cv2.imwrite(r'C:\Users\14999\Desktop\2\label{}_{}.png'.format(i,j),label)
        j+=1
        if j>10:
            break
##########################################################################################
#data改名
# for i in range(1,801):
#     path = io.ImageCollection(r'C:\Users\14999\Desktop\1\data{}_*.png'.format(i))
#     j = 6
#     for path1 in path:
#         cv2.imwrite(r'C:\Users\14999\Desktop\data\data{}_{}.png'.format(i,j),path1)
#         j+=1
###########################################################################################
#检查
# for i in range(1,801):
#     path = io.ImageCollection(r'C:\Users\14999\Desktop\data\data{}_*.png'.format(i))
#     if(len(path)!=11):
#         print(i)
