import cv2
import numpy as np
from scipy.sparse.linalg import spsolve


RESULT_PATH = 'results/'


def _reshape_img(img, shape, offset):
    new_img = np.zeros(shape, dtype=img.dtype)
    if len(shape) == 3:
        new_img[offset[0]:img.shape[0] + offset[0], offset[1]:img.shape[1] + offset[1], :] = img
    else:
        new_img[offset[0]:img.shape[0] + offset[0], offset[1]:img.shape[1] + offset[1]] = img
    return new_img


def _get_mask_indices(fixed_mask):
    mask_indices = {}
    cnt = 0
    for y in range(fixed_mask.shape[0]):
        for x in range(fixed_mask.shape[1]):
            if fixed_mask[y, x] > 127:
                mask_indices[(y, x)] = cnt
                cnt += 1
    return mask_indices


def _get_A_and_b(source, target, mask_indices):
    A = np.zeros((len(mask_indices), len(mask_indices)), dtype=int)
    b = np.zeros((len(mask_indices), 3), dtype=int)

    for pixel in mask_indices:
        idx = mask_indices[pixel]

        A[idx, idx] = 4
        b[idx] += (4 * source[pixel])

        for y_move, x_move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            moved_pixel = (pixel[0]+y_move, pixel[1]+x_move)

            b[idx] -= source[moved_pixel]

            if moved_pixel in mask_indices:
                A[idx, mask_indices[moved_pixel]] = -1
            else:
                b[idx] += target[moved_pixel]

    return A, b


def _fix_target(x, mask_indices, target):
    for pixel in mask_indices:
        idx = mask_indices[pixel]

        for d in range(channels):
            x[idx][d] = max(0, x[idx][d])
            x[idx][d] = min(255, x[idx][d])

            target[pixel][d] = np.uint8(x[idx][d])


for i in range(1, 3):
    mask = cv2.imread("bleding_mask%d.jpg" % i, 0)
    source = cv2.imread("bleding_source%d.jpg" % i)
    target = cv2.imread("bleding_target%d.jpg" % i)

    h, w, channels = target.shape
    hs, ws, _ = source.shape

    fixed_source = _reshape_img(source, target.shape, [h-h//10-hs, w//10])
    fixed_mask = _reshape_img(mask, (target.shape[0], target.shape[1]), [h-h//10-hs, w//10])

    mask_indices = _get_mask_indices(fixed_mask)

    A, b = _get_A_and_b(fixed_source.astype('int'), target.astype('int'), mask_indices)

    x = spsolve(A, b)

    _fix_target(x, mask_indices, target)

    cv2.imwrite(RESULT_PATH + "bleding_result%d.jpg" % i, target)
