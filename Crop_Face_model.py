# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt


# reading an image
im = cv2.imread('Bill_Gates.jpg')
height, width, depth = im.shape
print(height, width, depth)
cv2.imshow('original',im)
cv2.waitKey(0)

# Filtering example
kernel = np.ones((5,5),np.float32)/25
im_filt = cv2.filter2D(im,-1,kernel)
cv2.imshow('filtered',im_filt)
cv2.waitKey(0)

# Crop example
#
im_cropped = im[20:120, 40:160]
cv2.imshow('cropped image',im_cropped)
cv2.waitKey(0)

# Profile plot example
x0 = width//2
profil_vert = im[:,x0,1]
plt.plot(profil_vert)
plt.ylabel('vertical profile')
plt.show()
im[:,x0] = [0,0,255]
cv2.imshow('image with line',im)
cv2.waitKey(0)

cv2.destroyAllWindows()