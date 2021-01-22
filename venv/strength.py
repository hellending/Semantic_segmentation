import cv2 as cv
import numpy as np
img = cv.imread(r"C:\Users\14999\Desktop\jjj\pre1.png", 0)
Imin, Imax = cv.minMaxLoc(img)[:2]
# 使用numpy计算
# Imax = np.max(img)
# Imin = np.min(img)
Omin, Omax = 0, 255
# 计算a和b的值
a = float(Omax - Omin) / (Imax - Imin)
b = Omin - a * Imin
out = a * img + b
out = out.astype(np.uint8)

cv.imwrite(r'C:\Users\14999\Desktop\jjj\pre2.png',out)
