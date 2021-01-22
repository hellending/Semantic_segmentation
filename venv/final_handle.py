import tensorflow.keras as tk
import cv2
img = tk.preprocessing.image.load_img(r'C:\Users\14999\Desktop\jjj\pre1.png')
img = tk.preprocessing.image.img_to_array(img)
for k in range(3):
    for i in range(128):
        for j in range(128):
            if img[i,j,k]<100:
                img[i,j,k] = 0
            else: img[i,j,k] = 255
cv2.imwrite(r'C:\Users\14999\Desktop\jjj\pre1.png',img)
