import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def log_fft(image):
    amplitude_image = np.absolute(image)
    log_amplitude_image = np.real(np.log(amplitude_image + np.ones(image.shape)))
    return log_amplitude_image


def _get_center(x):
    center = int(x / 2)
    if x % 2 != 0:
        center += 1
    return center


def get_gaussian_filter(x_len,  y_len, sigma):
    center_y =_get_center(y_len)
    center_x = _get_center(x_len)
    x, y = np.meshgrid(np.linspace(-1.0 * center_y, center_y, y_len), np.linspace(-1.0 * center_x, center_x, x_len))
    d =x * x + y * y
    g = np.exp(-(d / (2.0 * sigma ** 2)))

    return g


def show_fourier(image, name):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

    ax1.imshow(image[:, :, 0], aspect='auto')
    ax1.set_title('b channel')

    ax2.imshow(image[:, :, 1], aspect='auto')
    ax2.set_title('g channel')

    ax3.imshow(image[:, :, 2], aspect='auto')
    ax3.set_title('r channel')

    plt.tight_layout()
    plt.savefig(name)


def show_gaussian(gaussian, name):
    h, w = gaussian.shape
    h = h / w * 8
    w = 8
    # w = w/h * 8
    # h = 8
    fig, (ax1) = plt.subplots(nrows=1, figsize=(w, h))

    ax1.imshow(gaussian, aspect='auto')
    ax1.set_title('gaussian')

    plt.tight_layout()
    plt.savefig(name)


def _apply_fourier(rgb_image, gaussian_filter, name):
    filtered_rgb_image_fft = np.zeros(rgb_image.shape, dtype='complex128')
    shifted_rgb_image = np.zeros(rgb_image.shape, dtype='complex128')
    for i in range(3):
        image = rgb_image[:,:, i]
        image_fft = np.fft.fft2(image)

        shifted_image = np.fft.fftshift(image_fft)
        shifted_rgb_image[:, :, i] = shifted_image

        filtered_image_fft = shifted_image * gaussian_filter
        filtered_rgb_image_fft[:, :, i] = filtered_image_fft

    if name == 'near':
        show_fourier(log_fft(shifted_rgb_image), 'Q4_05_dft_near.jpg')
    else:
        show_fourier(log_fft(shifted_rgb_image), 'Q4_06_dft_far.jpg')

    return filtered_rgb_image_fft


def _apply_cutoff(g_filter, cutoff):
    g_filter[g_filter < cutoff] = 0
    return g_filter


def get_low_pass_image(image, sigma, cutoff):
    gaussian_filter = get_gaussian_filter(image.shape[0], image.shape[1], sigma)
    show_gaussian(gaussian_filter, 'Q4_08_lowpass_%d.jpg' % sigma)

    gaussian_filter = _apply_cutoff(gaussian_filter, cutoff)
    show_gaussian(gaussian_filter, 'Q4_10_lowpass_cutoff.jpg')

    low_pass_img = _apply_fourier(image, gaussian_filter, 'far')
    show_fourier(log_fft(low_pass_img), 'Q4_12_lowpassed.jpg')

    return low_pass_img, gaussian_filter


def get_high_pass_image(image, sigma, cutoff):
    gaussian_filter = get_gaussian_filter(image.shape[0], image.shape[1], sigma)
    show_gaussian(1 - gaussian_filter, 'Q4_07_highpass_%d.jpg' % sigma)
    gaussian_filter = _apply_cutoff(gaussian_filter, cutoff)
    show_gaussian(1 - gaussian_filter, 'Q4_09_highpass_cutoff.jpg')

    gaussian_filter = 1 - gaussian_filter

    high_pass_img = _apply_fourier(image, gaussian_filter, 'near')
    show_fourier(log_fft(high_pass_img), 'Q4_11_highpassed.jpg')

    return high_pass_img, gaussian_filter


def output_vis(image, cnt=5):
    images = []
    h, w = image.shape[0], 0

    for i in range(cnt):
        tmp = cv2.resize(image, (0, 0), fx=0.5 ** i, fy=0.5 ** i)
        w += tmp.shape[1]
        images.append(tmp)

    stack = np.ones((h, w, 3)) * 255

    current_x = 0
    for img in images:
        stack[h - img.shape[0]:, current_x: img.shape[1] + current_x, :] = img
        current_x += img.shape[1]

    return stack


def _get_hybrid(low_pass_image, high_pass_image, low_pass_filter, high_pass_filter):
    cv2.imwrite("Q4_low_pass_img.jpg", _inverse(low_pass_image))
    cv2.imwrite("Q4_high_pass_img.jpg", _inverse(high_pass_image))
    mult_mtx = low_pass_filter * high_pass_filter
    low_pass_ratio = low_pass_filter + mult_mtx
    low_pass_ratio[low_pass_ratio > 1] = 0.5
    high_pass_ratio = high_pass_filter + mult_mtx
    high_pass_ratio[high_pass_ratio > 1] = 0.5
    hybrid = low_pass_image.copy()
    for i in range(3):
        hybrid[:, :, i] = low_pass_ratio * low_pass_image[:, :, i] + high_pass_ratio * high_pass_image[:, :, i]

    show_fourier(log_fft(hybrid), 'Q4_13_hybrid_frequency.jpg')
    hybrid_image = _inverse(hybrid)

    return hybrid_image


def _inverse(filtered_image_fft):
    filtered_rgb_image = np.zeros(filtered_image_fft.shape, dtype='float64')
    for i in range(3):
        filtered_image_ishifted = np.fft.ifftshift(filtered_image_fft[:, :, i])
        filtered_image = np.fft.ifft2(filtered_image_ishifted)
        filtered_image = np.real(filtered_image)
        filtered_rgb_image[:, :, i] = filtered_image
    return filtered_rgb_image


def _get_matrix_transfrom(point1, point2, point3, pointp1, pointp2, pointp3):
    # a x1 + b y1 + tx = xp1
    A = np.array([[point1[0], point1[1], 1], [point2[0], point2[1], 1], [point3[0], point3[1], 1]])
    B = np.array([pointp1[0], pointp2[0], pointp3[0]])
    a, b, tx = np.linalg.solve(A, B)

    # c x1 + d y1 + ty = yp1
    A = np.array([[point1[0], point1[1], 1], [point2[0], point2[1], 1], [point3[0], point3[1], 1]])
    B = np.array([pointp1[1], pointp2[1], pointp3[1]])
    c, d, ty = np.linalg.solve(A, B)

    return np.array([[a, b, tx], [c, d, ty]])


def _crop_img(img, crop_size):
    h, w = img.shape[0], img.shape[1]
    return img[crop_size: h-crop_size, crop_size: w-crop_size]


def _matche_images(near, near_points, far, far_points):
    w, h = near.shape[0], near.shape[1]

    M_near = _get_matrix_transfrom(near_points['left_eye'], near_points['right_eye'], near_points['nose'],
                                   far_points['left_eye'], far_points['right_eye'], far_points['nose'])

    affined_near = cv2.warpAffine(near, M_near, (h, w))
    affined_near = _crop_img(affined_near, 190)
    affined_far = _crop_img(far, 190)

    return affined_near, affined_far


if __name__ == "__main__":
    near = cv2.imread("Q4_01_near.jpg")
    near_points = {
        'left_eye': (418, 567),
        'right_eye': (573, 564),
        'nose': (501, 650)
    }

    far = cv2.imread("Q4_02_far.jpg")
    far_points = {
        'left_eye': (410, 454),
        'right_eye': (539, 459),
        'nose': (471, 533)
    }

    near, far = _matche_images(near, near_points, far, far_points)
    near = cv2.resize(near, (0, 0), fx=3, fy=3)
    far = cv2.resize(far, (0, 0), fx=3, fy=3)
    cv2.imwrite("Q4_03_near.jpg", near)
    cv2.imwrite("Q4_04_far.jpg", far)

    sigma_low, sigma_high = 30, 10
    cutoff_low, cutoff_high = 1/(2 * math.pi * sigma_low), 1/(2 * math.pi * sigma_high)

    low_pass_image, low_pass_filter = get_low_pass_image(far, sigma_low, cutoff_low)
    high_pass_image, high_pass_filter = get_high_pass_image(near, sigma_high, cutoff_high)

    hybrid_image = _get_hybrid(low_pass_image, high_pass_image, low_pass_filter, high_pass_filter)

    cv2.imwrite("Q4_14_hybrid_near.jpg", hybrid_image)
    cv2.imwrite("Q4_15_hybrid_far.jpg", cv2.resize(hybrid_image, (0, 0), fx=0.5 ** 4, fy=0.5 ** 4))
    cv2.imwrite("Q4_scaled.jpg", output_vis(hybrid_image))

