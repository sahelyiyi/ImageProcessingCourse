import cv2
import numpy as np
from random import randint


sample = cv2.imread('sample.jpg')
h, w, d = sample.shape
patch_scale = 4
patch_size = min(int(h/patch_scale), int(w/patch_scale))


scale = 7
texture = np.zeros((patch_size*scale, patch_size*scale, d))
for i in range(scale):
    for j in range(scale):
        random_y, random_x = randint(0, h - patch_size), randint(0, w - patch_size)
        random_patch = sample[random_y:random_y+patch_size, random_x:random_x+patch_size]
        texture[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = random_patch

cv2.imwrite('im1.jpg', texture)


