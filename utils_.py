#################
# 在指定目录下放置需要水平不正的图片文件，注意文件夹和文件名不能有中文
# 图像源文件在 文件夹\in 目录，输入为 文件夹\out 目录
# 注意：源文件夹里没有其他文件或文件夹，没有做相关判断
# 通过霍夫变换（网上找的）
# 运行后，在另一个文件夹中输出调整好方向的文件
#################

import os
import cv2
import math
import numpy as np
from scipy import ndimage
import imageio

file_path='Swin-Unet/data_zEI/images/image_196.jpg'
# 读取图像
img = cv2.imread(file_path)
# 二值化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 边缘检测
edges = cv2.Canny(gray,50,150,apertureSize = 3)

#霍夫变换，摘自https://blog.csdn.net/feilong_csdn/article/details/81586322
lines = cv2.HoughLines(edges,1,np.pi/180,0)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
if x1 == x2 or y1 == y2:
    t=0
else:
    t = float(y2-y1)/(x2-x1)
print( "t:   "+ str(t))
# 得到角度后
rotate_angle = math.degrees(math.atan(t))
if rotate_angle > 45:
    rotate_angle = -90 + rotate_angle
elif rotate_angle < -45:
    rotate_angle = 90 + rotate_angle
# 图像根据角度进行校正
rotate_img = ndimage.rotate(img, rotate_angle)
print( "jiaodu:   "+ str(rotate_angle))
# 输出图像
imageio.imwrite('utils_result.png', rotate_img)
print( " 已转换")

