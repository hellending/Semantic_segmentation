import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
model = tf.keras.models.load_model(r'C:\Users\14999\Desktop\model_last')
img = tf.keras.preprocessing.image.load_img(r'C:\Users\14999\Desktop\test\data.png')
img = tf.keras.preprocessing.image.img_to_array(img)/255.0
test = np.zeros((128,128,3),dtype=np.uint8)
test[:,:,:] = img[:,:,:]
predict = model.predict(np.expand_dims(test,axis=0),verbose=2)
# print(predict)
predict*=255
pre = predict[0].astype(np.uint8)
# print(pre)
cv2.imwrite(r'C:\Users\14999\Desktop\jjj\pre1.png',pre)