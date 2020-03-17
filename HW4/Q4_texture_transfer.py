import numpy as np
import cv2

SCALE = 0.65
RESULT_PATH = 'results/'


def get_best_match(texture_img, result_img, target_img, y, x, patch_size, match_size, err_threshold, itr_num):
    result_up = get_up(result_img, y, x, patch_size, match_size)

    result_left = get_left(result_img, y, x, patch_size, match_size)

    target_patch = target_img[y:y + patch_size, x:x + patch_size]
    existing_patch = result_img[y:y + patch_size, x:x + patch_size]

    img_height, img_width, _ = texture_img.shape
    margin_below, margin_right = result_left.shape[0], result_up.shape[1]
    y1, y2 = match_size, img_height - margin_below + 1
    x1, x2 = match_size, img_width - margin_right + 1

    err_map = np.full((img_height, img_width), np.inf)

    err_map[y1:y2, x1:x2] = 2 * cv2.matchTemplate(texture_img[match_size:, match_size:], target_patch, cv2.TM_SQDIFF)
    if itr_num > 0:
        err_map[y1:y2, x1:x2] += cv2.matchTemplate(texture_img[match_size:, match_size:], existing_patch, cv2.TM_SQDIFF)
    if y > 0:
        err_map[y1:y2, x1:x2] += 2 * cv2.matchTemplate(texture_img[:-margin_below, match_size:], result_up, cv2.TM_SQDIFF)
    if x > 0:
        err_map[y1:y2, x1:x2] += 2 * cv2.matchTemplate(texture_img[match_size:, :-margin_right], result_left, cv2.TM_SQDIFF)

    p_choices = np.where(err_map <= np.min(err_map) * (1.0 + err_threshold))
    if len(p_choices[0]) == 0:
        return np.unravel_index(np.argmin(err_map), err_map.shape)
    return np.random.choice(p_choices[0]), np.random.choice(p_choices[1])


def get_up(img, y, x, dx, match_size):
    return img[y - match_size:y, x:x + dx]


def get_left(img, y, x, dy, match_size):
    return img[y:y + dy, x - match_size:x]


def _get_mask(diff):
    min_indices = []
    error = [list(diff[0])]
    for idx1 in range(1, diff.shape[0]):
        e = [np.inf] + error[-1] + [np.inf]
        min_index = []
        min_values = []
        for idx2 in range(1, len(e) - 1):
            min_value = e[idx2]
            min_idx = 0
            for idx3 in range(-1, 2):
                if e[idx2 + idx3] < min_value:
                    min_value = e[idx2 + idx3]
                    min_idx = idx3
            min_index.append(min_idx)
            min_values.append(min_value)
        min_indices.append(min_index)
        error.append(list(diff[idx1] + min_values))

    path = []
    min_index = np.argmin(error[-1])
    path.append(min_index)

    min_indices.reverse()
    for idx in min_indices:
        min_index = min_index + idx[min_index]
        path.append(min_index)

    path.reverse()
    mask = np.zeros((diff.shape[0], diff.shape[1], 3))
    for i in range(len(path)):
        mask[i, :path[i] + 1] = 1

    return mask


def fade_left(texture_match, result_match):
    diff = ((texture_match - result_match) ** 2).mean(2)
    mask = _get_mask(diff)

    texture_match = texture_match * mask + result_match * (1 - mask)
    texture_match = texture_match.astype('float32')
    return texture_match


def fade_up(texture_match, result_match):
    diff = ((texture_match - result_match) ** 2).mean(2).T
    mask = _get_mask(diff).transpose(1, 0, 2)

    texture_match = texture_match * mask + result_match * (1 - mask)
    texture_match = texture_match.astype('float32')
    return texture_match


def texture_transfer(texture_img, target_img, patch_size, match_size, result_file_name, err_threshold=0.0, n=5):
    out_height, out_width, _ = target_img.shape
    result = np.zeros(target_img.shape).astype(np.float32)
    for itr_num in range(n):
        for y in range(0, out_height, patch_size):
            for x in range(0, out_width, patch_size):
                best_y, best_x = get_best_match(texture_img, result, target_img,y, x, patch_size, match_size, err_threshold, itr_num)

                dy, dx, _ = result[y:y + patch_size, x:x + patch_size].shape
                result[y:y + dy, x:x + dx] = texture_img[best_y:best_y + dy, best_x:best_x + dx]

                if y > 0:
                    texture_up = get_up(texture_img, best_y, best_x, dx, match_size)
                    up_patch = get_up(result, y, x, dx, match_size)
                    result[y - match_size:y, x:x + dx] = fade_up(texture_up, up_patch)
                if x > 0:
                    texture_left = get_left(texture_img, best_y, best_x, dy, match_size)
                    left_patch = get_left(result, y, x, dy, match_size)
                    result[y:y + dy, x - match_size:x] = fade_left(texture_left, left_patch)

        patch_size = max(1, int(patch_size * SCALE))
        match_size = max(1, int(match_size * SCALE))
        err_threshold *= 0.5

        if itr_num < n - 1:
            cv2.imwrite(RESULT_PATH + result_file_name[:-4] + '_iter%d.jpg' % (itr_num + 1), result)
    cv2.imwrite(RESULT_PATH + result_file_name, result)


texture_img = cv2.imread('snake_texture.jpg').astype(np.float32)
target_img = cv2.imread('cat.jpg').astype(np.float32)

texture_img_height, texture_img_width, _ = texture_img.shape
patch_size = int(min(texture_img_height, texture_img_width) / 3)
match_size = int(patch_size / 3)

texture_transfer(texture_img, target_img, patch_size, match_size, 'im4.jpg')
