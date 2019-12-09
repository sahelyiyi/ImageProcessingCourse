import cv2
import numpy as np
from scipy import misc

from collections import defaultdict


def get_pixel_by_values(img):
    h, w = img.shape
    res = defaultdict(list)

    for y in range(h):
        for x in range(w):
            res[img[y, x]].append((y, x))

    return res


def divide_hsv(source, target_hist):
    source = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
    value = source[:,:,2]

    pixel_values = get_pixel_by_values(value)

    counter = 0
    times = 0

    for pixel_value, pixels in sorted(pixel_values.items(), key = lambda kv:(kv[1], kv[0])):
        for y, x in pixels:
            while times >= target_hist[counter]:
                counter += 1
                times = 0
            source[y, x, 2] = counter
            times += 1

    noise_target = cv2.cvtColor(source, cv2.COLOR_HSV2RGB)
    small = misc.imresize(noise_target, 0.05)
    cv2.imwrite('matched_hsv.jpg', small)
    cv2.imwrite('im02.jpg', noise_target)


def divide(source, target_hist):

    pixel_values = get_pixel_by_values(source)

    counter = 0
    times = 0

    for i in range(256):
        pixels = pixel_values[i]
        for y, x in pixels:
            while times >= target_hist[counter]:
                counter += 1
                times = 0
            source[y, x] = counter
            times += 1

    return source


def map_hist(img_hist, target_hist):
    img_hist = np.cumsum(img_hist)
    target_hist = np.cumsum(target_hist)

    mapping = []
    counter = 0
    for i in range(len(img_hist)):
        while counter < len(img_hist)-1 and target_hist[counter] < img_hist[i]:
            counter += 1
        if counter == 0:
            mapping.append(0)
        else:
            if target_hist[counter] - img_hist[i] < img_hist[i] - target_hist[counter-1]:
                mapping.append(counter)
            else:
                mapping.append(counter-1)
                counter -= 1
    return mapping


def _one_channel_normal(source, target_hist):
    img_hist = cv2.calcHist([source], [0], None, [256], [0, 256])

    map = map_hist(img_hist, target_hist)

    h, w = source.shape
    for y in range(h):
        for x in range(w):
            source[y, x] = map[source[y, x]]

    return source


def normal(source, target_hist):
    b, g, r = source[:, :, 0], source[:, :, 1], source[:, :, 2]

    b = _one_channel_normal(b, target_hist)
    g = _one_channel_normal(g, target_hist)
    r = _one_channel_normal(r, target_hist)

    source[:, :, 0], source[:, :, 1], source[:, :, 2] = b, g, r

    small = misc.imresize(source, 0.05)
    cv2.imwrite('matched_normal.jpg', small)


def _get_inputs():
    source = cv2.imread('IMG_2919.JPG')
    h, w, _ = source.shape
    mean = int(h * w / 255)
    remain = h * w - mean * 255
    target_hist = np.full(255, mean)
    target_hist[-1] += remain
    return source, target_hist


def divide_rgb(source, target_hist):
    b, g, r = source[:, :, 0], source[:, :, 1], source[:, :, 2]

    b = divide(b, target_hist)
    g = divide(g, target_hist)
    r = divide(r, target_hist)

    source[:, :, 0], source[:, :, 1], source[:, :, 2] = b, g, r

    small = misc.imresize(source, 0.05)
    cv2.imwrite('matched_divide.jpg', small)


source, target_hist = _get_inputs()
divide_rgb(source, target_hist)


source, target_hist = _get_inputs()
normal(source, target_hist)

source, target_hist = _get_inputs()
divide_hsv(source, target_hist)
