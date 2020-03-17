import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
from collections import defaultdict

image = cv2.imread('IMG_2805.JPG')
image = cv2.imread('IMG_7760E8D6E09D-1.jpeg')
h, w, _ = image.shape

flat_image = np.reshape(image, [-1, 3])

bandwidth2 = estimate_bandwidth(flat_image,
                                quantile=.2, n_samples=500)
mean_shift = MeanShift(bandwidth2, bin_seeding=True)
mean_shift.fit(flat_image)
labels = mean_shift.labels_
labels = np.reshape(labels, [h, w])

label_values = defaultdict(list)
for y in range(h):
    for x in range(w):
        label_values[labels[y, x]].append(image[y, x])

mean_values = {}
for label in label_values:
    label_values[label] = np.array(label_values[label]).astype('uint8')
    mean_values[label] = np.mean(label_values[label], 0).astype('uint8')

new_image = image.copy()
for y in range(h):
    for x in range(w):
        new_image[y, x] = mean_values[labels[y, x]]


#cv2.imwrite('im05.jpg', new_image)
cv2.imwrite('saleh2.jpg', new_image)
