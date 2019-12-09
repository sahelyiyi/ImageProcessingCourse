import cv2
import numpy as np


def power_law_tranformation(img_original, gamma):
    power_law_function = lambda i: 255.0 * pow(i / 255.0, gamma)
    vectorized = np.vectorize(power_law_function)
    res = vectorized(img_original)
    return res


original_img = cv2.imread('im030.jpg')

edited_imgs = []
scale = 0.05
for gamma in range(1, 11):
    new_img = power_law_tranformation(np.copy(original_img), gamma=gamma / 10.0)
    new_h, new_w = new_img.shape[1] * scale, new_img.shape[0] * scale
    new_img = cv2.resize(new_img, (int(new_h), int(new_w)))
    edited_imgs.append(new_img)

final1 = np.concatenate(edited_imgs[0:5], axis=0)
final2 = np.concatenate(edited_imgs[5:10], axis=0)
final = np.concatenate((final1, final2), axis=1)

cv2.imwrite('final01.jpg', final)

best_gamma = 0.55
best_result = power_law_tranformation(np.copy(original_img), gamma=best_gamma)
cv2.imwrite('im01.jpg', best_result)

