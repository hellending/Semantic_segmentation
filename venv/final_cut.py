import cv2
import tensorflow as tf
import numpy as np
path = r'C:\Users\14999\Desktop\data_final1\test.png'
img = tf.keras.preprocessing.image.load_img(path)
img = tf.keras.preprocessing.image.img_to_array(img)
h = np.zeros((128,128,3))
h[:116,:88,:] = img[384:,512:,:]
cv2.imwrite(r'C:\Users\14999\Desktop\data_final1\test20.png',h)

