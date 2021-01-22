import tensorflow as tf
import numpy as np
import cv2
path = r'C:\Users\14999\Desktop\work_out1'
img1 = tf.keras.preprocessing.image.load_img(path+'\\'+'final1.png')
img1 = tf.keras.preprocessing.image.img_to_array(img1)
img2 = tf.keras.preprocessing.image.load_img(path+'\\'+'final2.png')
img2 = tf.keras.preprocessing.image.img_to_array(img2)
img3 = tf.keras.preprocessing.image.load_img(path+'\\'+'final3.png')
img3 = tf.keras.preprocessing.image.img_to_array(img3)
img4 = tf.keras.preprocessing.image.load_img(path+'\\'+'final4.png')
img4 = tf.keras.preprocessing.image.img_to_array(img4)
img5 = tf.keras.preprocessing.image.load_img(path+'\\'+'final5.png')
img5 = tf.keras.preprocessing.image.img_to_array(img5)
img6 = tf.keras.preprocessing.image.load_img(path+'\\'+'final6.png')
img6 = tf.keras.preprocessing.image.img_to_array(img6)
img7 = tf.keras.preprocessing.image.load_img(path+'\\'+'final7.png')
img7 = tf.keras.preprocessing.image.img_to_array(img7)
img8 = tf.keras.preprocessing.image.load_img(path+'\\'+'final8.png')
img8 = tf.keras.preprocessing.image.img_to_array(img8)
img9 = tf.keras.preprocessing.image.load_img(path+'\\'+'final9.png')
img9 = tf.keras.preprocessing.image.img_to_array(img9)
img10 = tf.keras.preprocessing.image.load_img(path+'\\'+'final10.png')
img10 = tf.keras.preprocessing.image.img_to_array(img10)
img11 = tf.keras.preprocessing.image.load_img(path+'\\'+'final11.png')
img11 = tf.keras.preprocessing.image.img_to_array(img11)
img12 = tf.keras.preprocessing.image.load_img(path+'\\'+'final12.png')
img12 = tf.keras.preprocessing.image.img_to_array(img12)
img13 = tf.keras.preprocessing.image.load_img(path+'\\'+'final13.png')
img13 = tf.keras.preprocessing.image.img_to_array(img13)
img14 = tf.keras.preprocessing.image.load_img(path+'\\'+'final14.png')
img14 = tf.keras.preprocessing.image.img_to_array(img14)
img15 = tf.keras.preprocessing.image.load_img(path+'\\'+'final15.png')
img15 = tf.keras.preprocessing.image.img_to_array(img15)
img16 = tf.keras.preprocessing.image.load_img(path+'\\'+'final16.png')
img16 = tf.keras.preprocessing.image.img_to_array(img16)
img17 = tf.keras.preprocessing.image.load_img(path+'\\'+'final17.png')
img17 = tf.keras.preprocessing.image.img_to_array(img17)
img18 = tf.keras.preprocessing.image.load_img(path+'\\'+'final18.png')
img18 = tf.keras.preprocessing.image.img_to_array(img18)
img19 = tf.keras.preprocessing.image.load_img(path+'\\'+'final19.png')
img19 = tf.keras.preprocessing.image.img_to_array(img19)
img20 = tf.keras.preprocessing.image.load_img(path+'\\'+'final20.png')
img20 = tf.keras.preprocessing.image.img_to_array(img20)

img = np.zeros((500,600,3))
img[0:128,0:128,:] = img1[:,:,:]
img[128:256,0:128,:] = img2[:,:,:]
img[256:384,0:128,:] = img3[:,:,:]
img[384:,0:128,:] = img4[:116,:,:]
img[0:128,128:256,:] = img5[:,:,:]
img[128:256,128:256,:] = img6[:,:,:]
img[256:384,128:256,:] = img7[:,:,:]
img[384:,128:256,:] = img8[:116,:,:]
img[0:128,256:384,:] = img9[:,:,:]
img[128:256,256:384,:] = img10[:,:,:]
img[256:384,256:384,:] = img11[:,:,:]
img[384:,256:384,:] = img12[:116,:,:]
img[0:128,384:512,:] = img13[:,:,:]
img[128:256,384:512,:] = img14[:,:,:]
img[256:384,384:512,:] = img15[:,:,:]
img[384:,384:512,:] = img16[:116,:,:]
img[0:128,512:,:] = img17[:,:88,:]
img[128:256,512:,:] = img18[:,:88,:]
img[256:384,512:,:] = img19[:,:88,:]
img[384:,512:,:] = img20[:116,:88,:]

cv2.imwrite(r'C:\Users\14999\Desktop\test_answer1.png',img)
