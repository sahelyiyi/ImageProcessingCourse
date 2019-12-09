import cv2
import numpy as np
from scipy import misc


def _match_imgs(img1, img2, img_name=None):
    # img2 is fixed and we move img1
    selected_sample = 500
    center = 1000
    common2 = img2[center:center+selected_sample, center:center+selected_sample]

    min_value = None
    min_diff_y = None
    min_diff_x = None
    for y in range(-100, 100):
        for x in range(-100, 100):
            common1 = img1[center+y:center+y+selected_sample, center+x:center+x+selected_sample]
            s = np.sum(abs(common1 - common2))

            if min_value is None or s < min_value:
                min_value = s
                min_diff_y = y
                min_diff_x = x

    if img_name:
        best_common1 = img1[center+min_diff_y:center+min_diff_y+selected_sample, center+min_diff_x:center+min_diff_x+selected_sample]
        small = misc.imresize(np.concatenate((common2, best_common1), axis=1), 0.2)
        cv2.imwrite(img_name, small)

    return min_diff_y, min_diff_x


aggregated_imgs = cv2.imread('BW.tif', 0)

img_size = int(aggregated_imgs.shape[0]/3)
imgs = []
for i in range(1, 4):
    imgs.append(aggregated_imgs[(i-1)*img_size:i*img_size, :])


dif_y1, dif_x1 = _match_imgs(np.copy(imgs[0]), np.copy(imgs[1]), 'best_match10.jpg')
dif_y2, dif_x2 = _match_imgs(np.copy(imgs[2]), np.copy(imgs[1]), 'best_match12.jpg')

# 41 -16
# -60 -3

center = max(abs(dif_y1), abs(dif_x1), abs(dif_y2), abs(dif_x2))
h, w = imgs[0].shape
new_h, new_w = h-2*center, w-2*center

b = imgs[0][center+dif_y1:center+dif_y1+new_h, center+dif_x1:center+dif_x1+new_w]
g = imgs[1][center:center+new_h, center:center+new_w]
r = imgs[2][center+dif_y2:center+dif_y2+new_h, center+dif_x2:center+dif_x2+new_w]


new_img = cv2.merge((b, g, r))

cv2.imwrite('im03.jpg', new_img)
