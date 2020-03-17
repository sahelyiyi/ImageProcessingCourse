import cv2
import numpy as np
import copy
import math
import os
import re


_ALPHA = 0.0001
_BETA = 0
_W_LINE = 1000
_W_EDGE = 100
_GAMMA = 100000

NEIGHBORS1 = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])
NEIGHBORS2 = np.array([[i*2, j*2] for i in range(-2, 3) for j in range(-2, 3)])

MIN_DIS = 20


def get_dis(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def _get_mean_dis(snake):
    snake_len = len(snake)
    distances = []
    for i in range(snake_len - 1, -1, -1):
        current_position = i
        next_position = (i + 1) % snake_len
        distances.append(get_dis(snake[current_position], snake[next_position]))
    mean_dis = np.mean(distances)
    return mean_dis


def get_internal_energy(snake):
    mean_dis = _get_mean_dis(snake)
    snake_len = len(snake)

    energy = 0
    for index in range(snake_len - 1, -1, -1):
        current_point = index % snake_len
        next_point = (index + 1) % snake_len
        prev_point = (index - 1) % snake_len

        x = np.sum((snake[next_point] - snake[current_point] - 2.0/3*mean_dis) ** 2)

        y = np.sum((snake[next_point] - 2 * snake[current_point] + snake[prev_point]) ** 2)

        energy += (_ALPHA * x + _BETA * y)
    return energy


def get_external_energy(gradient, image, snake):
    snake_len = len(snake)

    pixel = 0
    for index in range(snake_len - 1):
        point = snake[index]
        pixel = +(image[point[1]][point[0]])
    pixel *= 255

    edge = 0
    for index in range(snake_len - 1):
        point = snake[index]
        edge = edge + ((gradient[point[1]][point[0]]))

    energy = _W_LINE * pixel - _W_EDGE * edge

    return energy


def get_total_energy(grediant, image, snake):
    internal_energy = get_internal_energy(snake)
    external_energy = get_external_energy(grediant, image, snake)

    energy = internal_energy + _GAMMA * external_energy

    return energy


def draw_contour(image, snake):
    new_img = np.copy(image)

    for s in snake:
        cv2.drawMarker(new_img, (s[0], s[1]), (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=5, thickness=2,
                       line_type=cv2.LINE_AA)

    return new_img


def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


def get_image_gradient(image):
    g_x = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=17))
    g_x = interval_mapping(g_x, np.min(g_x), np.max(g_x), 0, 255)

    g_y = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=17))
    g_y = interval_mapping(g_y, np.min(g_y), np.max(g_y), 0, 255)

    gradient = g_x**2 + g_y**2

    return gradient


def check_point(image, point):
    return np.all(point < np.shape(image)) and np.all(point > 0)


def get_snake(center, radius, num_points):

    points = np.zeros((num_points, 2), dtype=np.int32)
    for i in range(num_points):
        theta = float(i) / num_points * (2 * np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        p = [x, y]
        points[i] = p

    return points


def double_snake(snake):
    mean_dis = _get_mean_dis(snake)
    print (mean_dis)
    new_snake = []
    snake_len = len(snake)
    for i in range(snake_len):
        current_node = snake[i]
        new_snake.append(current_node)
        next_node = snake[(i+1) % snake_len]
        if get_dis(current_node, next_node) > MIN_DIS:
            mid_node = (current_node + next_node)/2
            new_snake.append(mid_node)
    return np.array(new_snake).astype('int32')


def active_contour(image_file, center, radius):
    image = cv2.imread(image_file, 0)

    snake = get_snake(center, radius, 30)

    gradient = get_image_gradient(image)

    snake_copy = copy.deepcopy(snake)

    NEIGHBORS = NEIGHBORS2
    cnt = 0
    x = 30
    for iter_num in range(150):

        if iter_num != 0 and iter_num % x == 0:
            snake = double_snake(snake)
            snake_copy = copy.deepcopy(snake)
            cnt += 1
            print (iter_num, x)
            if cnt > 2:
                NEIGHBORS = NEIGHBORS1
            if x > 10:
                x = int(x/1.25)

        for point_index, point in enumerate(snake):

            min_energy = float("inf")

            for move_index, move in enumerate(NEIGHBORS):
                new_point = (point + move)

                if not check_point(image, new_point):
                    continue
                if not check_point(image, point):
                    continue

                snake_copy[point_index] = new_point
                energy = get_total_energy(gradient, image, snake_copy)

                if energy < min_energy:
                    min_energy = energy
                    min_index = move_index

            new_point = snake[point_index] + NEIGHBORS[min_index]
            snake[point_index] = new_point

        if iter_num % 3 == 0:
            if iter_num % 15 == 0:
                print (iter_num)
            new_img = draw_contour(image, snake)
            cv2.imwrite('images/contour%d.jpg' % iter_num, new_img)

        snake_copy = copy.deepcopy(snake)


def atoi(text):
    return int(text) if text.isdigit() else text


def sort_image_names(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    center = (400, 380)
    radius = 210
    active_contour("edge.jpg", center, radius)

    image_folder = 'images'
    video_name = 'movie01.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=sort_image_names)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

