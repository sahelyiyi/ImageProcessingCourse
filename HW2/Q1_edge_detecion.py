import scipy.ndimage as ndi
import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d

sigma = 3


def show_matrix(x, y, g, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, g, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    plt.savefig('Q1-%s.jpg' % name)


def get_gaussian_filter(sigma):
    x, y = np.meshgrid(np.linspace(-3*sigma, 3*sigma, 6*sigma), np.linspace(-3*sigma, 3*sigma, 6*sigma))
    d =x * x + y * y
    g = 1.0/(2.0 * math.pi * (sigma**2)) * np.exp(-(d / (2.0 * sigma ** 2)))

    # g = g / g.sum()

    show_matrix(x, y, g, '%d-gaussian' % sigma)

    return x, y, g


def separate_matrix(m, prefix):
    u, s, v = np.linalg.svd(m, full_matrices=True)
    u = u[:, 0] * math.sqrt(s[0])
    plt.figure()
    plt.plot(u)
    plt.savefig('Q1-u%s.jpg' % prefix)
    u = u.reshape(u.shape[0], 1)

    v = v[0, :] * math.sqrt(s[0])
    plt.figure()
    plt.plot(v)
    plt.savefig('Q1-v%s.jpg' % prefix)
    v = v.reshape(1, v.shape[0])
    return u, v


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
    show_matrix(x, y, dx, 'dx')
    show_matrix(x, y, dy, 'dy')
    ux, vx = separate_matrix(dx, 'x')
    uy, vy = separate_matrix(dy, 'y')

    img = cv2.imread('books.jpg', 0)

    ver_row = convolve2d(img, ux, 'valid')
    cv2.imwrite('Q1-02-ver-row.jpg', abs(ver_row))

    ver_col = convolve2d(img, vx, 'valid')
    cv2.imwrite('Q1-04-ver-col.jpg', abs(ver_col))

    hor_row = convolve2d(img, uy, 'valid')
    cv2.imwrite('Q1-01-hor-row.jpg', abs(hor_row))

    hor_col = convolve2d(img, vy, 'valid')
    cv2.imwrite('Q1-03-hor-col.jpg', abs(hor_col))

    divided_ver = convolve2d(img, convolve2d(ux, vx), 'valid')
    cv2.imwrite('Q1-separated-ver.jpg', abs(divided_ver))
    cv2.imwrite('Q1-scaled-separated-ver.jpg', abs(divided_ver*10))
    ver = convolve2d(img, dx, 'valid')
    print ('distance between separated and original vertical is', _get_dis(divided_ver, ver))
    cv2.imwrite('Q1-06-ver.jpg', abs(ver))
    cv2.imwrite('Q1-scaled-ver.jpg', abs(ver*10))

    divided_hor = convolve2d(img, convolve2d(uy, vy), 'valid')
    cv2.imwrite('Q1-separated-hor.jpg', abs(divided_hor))
    cv2.imwrite('Q1-scaled-separated-hor.jpg', abs(divided_hor*10))
    hor = convolve2d(img, dy, 'valid')
    print ('distance between separated and original horizontal is', _get_dis(divided_hor, hor))
    cv2.imwrite('Q1-05-hor.jpg', abs(hor))
    cv2.imwrite('Q1-scaled-hor.jpg', abs(hor*10))

    grad_mag = calculate_grad_mag(ver, hor)
    cv2.imwrite('Q1-07-grad-mag.jpg', grad_mag)

    grad_dir = calculate_gard_dir(ver, hor)
    cv2.imwrite('Q1-08-grad-dir.jpg', grad_dir)

    threshold = np.mean(grad_mag) * 3.5
    new_grad = np.zeros(grad_mag.shape)
    new_grad[grad_mag > threshold] = 255
    cv2.imwrite('Q1-09-edge.jpg', new_grad)
