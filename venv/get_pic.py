import numpy as np
import tensorflow as tf
from PIL import Image as Img
import os,cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def creat_x_database():
    dataSet = []
    labelSet = []
    url_data = r'C:\Users\14999\Desktop\test_final'
    list_data = os.listdir(url_data)
    url_label = r'C:\Users\14999\Desktop\label_final'
    list_label = os.listdir(url_label)
    for i in list_data:
        img = tf.keras.preprocessing.image.load_img(url_data+'\\'+i)
        img = tf.keras.preprocessing.image.img_to_array(img)
        dataSet.append(img)
    for i in list_label:
        img = tf.keras.preprocessing.image.load_img(url_label+'\\'+i,grayscale=True)
        img = tf.keras.preprocessing.image.img_to_array(img)
        labelSet.append(img)

    dataSet = np.array(dataSet,dtype=float)
    labelSet = np.array(labelSet,dtype=float)
    return dataSet,labelSet