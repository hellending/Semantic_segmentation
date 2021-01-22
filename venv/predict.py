import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
model = tf.keras.models.load_model(r'C:\Users\14999\Desktop\model1')
img1 = tf.keras.preprocessing.image.load_img(r'C:\Users\14999\Desktop\test_data\test1.png')
img2 = tf.keras.preprocessing.image.load_img(r'C:\Users\14999\Desktop\test_data\test2.png')
img1 = tf.keras.preprocessing.image.img_to_array(img1)/255.0
img2 = tf.keras.preprocessing.image.img_to_array(img2)/255.0
h1,w1,x = img1.shape
h2,w2,y = img2.shape
test1 = np.zeros((640,640,3),dtype=np.uint8)
test2 = np.zeros((640,640,3),dtype=np.uint8)
test1[0:h1,0:w1,:] = img1[:,:,:]
test2[0:h2,0:w2,:] = img2[:,:,:]
# pre1 = pre1[0:h1,0:w1,:]
# pre2 = pre2[0:h2,0:w2,:]
t1 = np.zeros((128,128,3))
t2 = np.zeros((128,128,3))
for i in range(0,5):
  for j in range(0,5):
      predict1 = model.predict(np.expand_dims(test1[128*i:128*(i+1),128*j:128*(j+1),:],axis=0),verbose=2)
      # predict1 = np.argmax(predict1, axis=1)
      predict2 = model.predict(np.expand_dims(test2[128*i:128*(i+1),128*j:128*(j+1),:],axis=0),verbose=2)
      # predict2 = np.argmax(predict2, axis=1)
      predict1*=255
      predict2*=255
      pre1 = predict1[0].astype(np.uint8)
      pre2 = predict2[0].astype(np.uint8)
      print(pre1)
      cv2.imwrite(r'C:\Users\14999\Desktop\jjj\pre1{}_{}.png'.format(i,j),pre1)
      cv2.imwrite(r'C:\Users\14999\Desktop\jjj\pre2{}_{}.png'.format(i,j),pre2)
# label = cv2.imread(r'C:\Users\14999\Desktop\label\label1_0.png')
# print(label)
# predict1 = model.predict(np.expand_dims(cv2.resize(img1,(128,128)),axis=0))
# predict2 = model.predict(np.expand_dims(cv2.resize(img2,(128,128)),axis=0))
# print(predict1)
# print(predict2)