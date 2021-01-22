import tensorflow.keras as tk
import cv2
path = r'C:\Users\14999\Desktop\label.png'
img = tk.preprocessing.image.load_img(path,grayscale=True)
img = tk.preprocessing.image.img_to_array(img)
for i in range(500):
    for j in range(600):
        if img[i,j]>0:
            img[i,j] = 0
        else: img[i,j] = 255
cv2.imwrite(r'C:\Users\14999\Desktop\label2.png',img)