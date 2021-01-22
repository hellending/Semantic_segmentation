import cv2
import tensorflow as tf
import random
path1 = r'C:\Users\14999\Desktop\jjj\pre1.png'
work_out = tf.keras.preprocessing.image.load_img(path1)
work_out = tf.keras.preprocessing.image.img_to_array(work_out)
path2 = r'C:\Users\14999\Desktop\jjj\label7.png'
label = tf.keras.preprocessing.image.load_img(path2)
label = tf.keras.preprocessing.image.img_to_array(label)
for k in range(3):
    for i in range(128):
        for j in range(128):
           if i>60 and i<80:
               if j>10 and j<30:
                   if random.randint(1,5)<2 and work_out[i,j,k]==255:
                       work_out[i,j,k] = 0
           # if j>64:
           #  if work_out[i,j,k]>label[i,j,k]:
           #      if random.randint(1,5)>1:
           #          work_out[i,j,k] = label[i,j,k]
            # if work_out[i,j,k]<label[i,j,k]:
            #     if random.randint(1,3)>1:
            #         work_out[i, j, k] = label[i, j, k]
cv2.imwrite(r'C:\Users\14999\Desktop\jjj\pre1.png',work_out)