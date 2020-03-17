import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.signal import convolve2d

sigma = 3


def get_gaussian_filter(sigma):
    x, y = np.meshgrid(np.linspace(-3*sigma, 3*sigma, 6*sigma), np.linspace(-3*sigma, 3*sigma, 6*sigma))
    d =x * x + y * y
    g = 1.0/(2.0 * math.pi * (sigma**2)) * np.exp(-(d / (2.0 * sigma ** 2)))

    return x, y, g


def calculate_grad_mag(ver, hor):
    ver_power = ver**2
    hor_power = hor**2
    return np.power(ver_power + hor_power, 0.5)


def calculate_gard_dir(ver, hor):
    return np.arctan(hor/ver)


def _get_dis(divided_item, item):
    return np.linalg.norm(item - divided_item)


if __name__ == "__main__":
    x, y, g = get_gaussian_filter(sigma)
    dx, dy = np.gradient(g)

    img = cv2.imread('tasbih.jpg', 0)

    ver = convolve2d(img, dx, 'valid')
    hor = convolve2d(img, dy, 'valid')

    grad_mag = calculate_grad_mag(ver, hor)

    grad_dir = calculate_gard_dir(ver, hor)

    threshold = np.mean(grad_mag) * 4
    new_grad = np.zeros(grad_mag.shape)
    new_grad[grad_mag > threshold] = 255
    new_grad[grad_mag < threshold] = 75
    cv2.imwrite('tasbih_edge.jpg', new_grad)
