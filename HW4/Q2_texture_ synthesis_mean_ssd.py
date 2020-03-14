import cv2
import numpy as np


def template_matching_ssd(src, patch_size, temp_left=None,  temp_up=None):
    h, w, d = src.shape
    if temp_left is not None:
        tl_h, tl_w, tl_d = temp_left.shape
    if temp_up is not None:
        tu_h, tu_w, tu_d = temp_up.shape

    score = np.full((h, w), 1e20)

    for dy in range(0, h-patch_size):
        for dx in range(0, w-patch_size):
            diff_l, diff_u = 0, 0
            if temp_left is not None and dy+tl_h < h and dx+tl_w < w:
                diff_l = ((src[dy:dy + tl_h, dx:dx + tl_w] - temp_left) ** 2).sum()
            if temp_up is not None and dy+tu_h < h and dx+tu_w < w:
                diff_u = ((src[dy:dy + tu_h, dx:dx + tu_w] - temp_up) ** 2).sum()
            score[dy, dx] = diff_l+diff_u

    pt = np.unravel_index(score.argmin(), score.shape)

    return pt[0], pt[1]


def get_left(texture, i, j, patch_size, match_size):
    return texture[i * patch_size:(i + 1) * patch_size, j * patch_size - match_size:j * patch_size]


def get_up(texture, i, j, patch_size, match_size):
    return texture[i * patch_size - match_size:i * patch_size, j * patch_size:(j + 1) * patch_size]


def fade_left(texture, i, j, patch_size, match_size, match_patch):
    tmp_left = get_left(texture, i, j, patch_size, match_size)
    tmp_left = (tmp_left + match_patch) / 2
    tmp_left = tmp_left.astype('float32')
    texture[i * patch_size:(i + 1) * patch_size, j * patch_size - match_size:j * patch_size] = tmp_left


def fade_up(texture, i, j, patch_size, match_size, match_patch):
    tmp_up = get_up(texture, i, j, patch_size, match_size)
    tmp_up = (tmp_up + match_patch) / 2
    tmp_up = tmp_up.astype('float32')
    texture[i * patch_size - match_size:i * patch_size, j * patch_size:(j + 1) * patch_size] = tmp_up


def get_best_match(sample, i, j, match_size, patch_size, tmp_left, tmp_up):
    if i == 0 and j == 0:
        return 1, 250
    elif j == 0:  # just up
        return template_matching_ssd(sample, patch_size + match_size, temp_left=None, temp_up=tmp_up)
    elif i == 0:  # just left
        return template_matching_ssd(sample, patch_size + match_size, temp_left=tmp_left, temp_up=None)
    else:
        return template_matching_ssd(sample, patch_size + match_size, temp_left=tmp_left, temp_up=tmp_up)


def generate_texture(scale, patch_size, match_size, d=3):
    texture = np.zeros((patch_size * scale, patch_size * scale, d)).astype('float32')
    for i in range(scale):
        for j in range(scale):
            tmp_left = get_left(texture, i, j, patch_size, match_size)
            tmp_up = get_up(texture, i, j, patch_size, match_size)
            best_y, best_x = get_best_match(sample, i, j, match_size, patch_size, tmp_left, tmp_up)

            if j > 0:
                left_patch = sample[best_y:best_y + patch_size, best_x:best_x + match_size]
                fade_left(texture, i, j, patch_size, match_size, left_patch)
                best_x += match_size
            if i > 0:
                up_patch = sample[best_y:best_y + match_size, best_x:best_x + patch_size]
                fade_up(texture, i, j, patch_size, match_size, up_patch)
                best_y += match_size

            best_patch = sample[best_y:best_y + patch_size, best_x:best_x + patch_size]
            texture[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = best_patch
    return texture


sample = cv2.imread('sample.jpg')
h, w, d = sample.shape
patch_scale = 4
patch_size = min(int(h/patch_scale), int(w/patch_scale))
scale = 7
match_size = 5
texture = generate_texture(scale, patch_size, match_size, d)
cv2.imwrite('im2.jpg', texture)
