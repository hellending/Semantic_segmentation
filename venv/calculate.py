import tensorflow.keras as tk
import os
path = r'C:\Users\14999\Desktop\资料\label\test2.png'
# list = os.listdir(path)
calculate = []
# for i in list:
    # path1 = path+'\\'+i
img = tk.preprocessing.image.load_img(path)
img = tk.preprocessing.image.img_to_array(img)
sum = 0
all = 500*600*3
for a in range(3):
    for b in range(500):
        for c in range(600):
            if img[b,c,a] == 255:
                sum+=1
calculate.append(float(sum/all))
print(calculate)

