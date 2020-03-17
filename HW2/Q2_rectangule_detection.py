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
from collections import defaultdict


INF = 1000000
sigma = 3
RESULT_PATH = 'results/'


def show_matrix(x, y, g, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, g, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    plt.savefig(RESULT_PATH + 'Q1-%s.jpg' % name)


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
    plt.savefig(RESULT_PATH + 'Q1-u%s.jpg' % prefix)
    u = u.reshape(u.shape[0], 1)

    v = v[0, :] * math.sqrt(s[0])
    plt.figure()
    plt.plot(v)
    plt.savefig(RESULT_PATH + 'Q1-v%s.jpg' % prefix)
    v = v.reshape(1, v.shape[0])
    return u, v


def calculate_grad_mag(ver, hor):
    ver_power = ver**2
    hor_power = hor**2
    return np.power(ver_power + hor_power, 0.5)


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def dis_points(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dis_slopes(l, new_l):
    s1 = slope(*l)
    s2 = slope(*new_l)
    return abs(s1 - s2)


def get_m_and_b(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def line_intersection(m1, b1, m2, b2):
    xi = (b1 - b2) / (m2 - m1)
    yi = m1 * xi + b1

    return int(xi), int(yi)


def _remove_outliers(uniques, lines):
    inliers = []
    for i in uniques:
        l = lines[i][0]
        min_dis = INF
        for j in uniques:
            if j == i:
                continue
            new_l = lines[j][0]
            dis_slope = dis_slopes(l, new_l)
            if dis_slope < min_dis:
                min_dis = dis_slope
        if min_dis < 1:
            inliers.append(i)

    return inliers


def _unique_lines(lines, img):
    uniques = []
    t_slope = 0.5
    t_dis = 50
    for i in range(len(lines)):
        l = lines[i][0]
        matched = False
        for item in uniques:
            new_l = lines[item][0]
            dis_slope = dis_slopes(l, new_l)
            min_dis = INF
            for k in range(2):
                for j in range(2):
                    _dis = dis_points(new_l[2 * k], new_l[2 * k + 1], l[2 * j], l[2 * j + 1])
                    if _dis < min_dis:
                        min_dis = _dis
            if dis_slope < t_slope and min_dis < t_dis:
                matched = True
                break
        if not matched:
            uniques.append(i)

    new_img = img.copy()
    for i in uniques:
        l = lines[i][0]
        cv2.line(new_img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1)
    cv2.imwrite(RESULT_PATH + 'Q2_uniques.jpg', new_img)

    uniques = _remove_outliers(uniques, lines)

    new_img = img.copy()
    for i in uniques:
        l = lines[i][0]
        cv2.line(new_img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1)
    cv2.imwrite(RESULT_PATH + 'Q2_inliers.jpg', new_img)

    return uniques


def _get_graph(uniques, lines):
    t = 10
    all_points = []
    line_points = defaultdict(list)
    point_lines = defaultdict(list)
    for i in uniques:
        l = lines[i][0]
        m1, b1 = get_m_and_b(*l)
        for j in uniques:
            if i == j:
                continue
            new_l = lines[j][0]
            m2, b2 = get_m_and_b(*new_l)
            x, y = line_intersection(m1, b1, m2, b2)
            min_dis_l = INF
            min_dis_new_l = INF
            for k in range(2):
                _dis = dis_points(new_l[2 * k], new_l[2 * k + 1], x, y)
                if _dis < min_dis_l:
                    min_dis_l = _dis
            for k in range(2):
                _dis2 = dis_points(l[2 * k], l[2 * k + 1], x, y)
                if _dis2 < min_dis_new_l:
                    min_dis_new_l = _dis2
            if min_dis_l < t or min_dis_new_l < t:
                if (x, y) in all_points:
                    continue
                all_points.append((x, y))
                point_index = len(all_points) - 1

                line_points[i].append(point_index)
                line_points[j].append(point_index)

                point_lines[point_index].append(i)
                point_lines[point_index].append(j)
    return all_points, line_points, point_lines


def _dfs_point(point_index, point_lines, line_points, check_points):
    check_points[point_index] = True

    points = []
    for line in point_lines[point_index]:
        for point_index2 in line_points[line]:
            if not check_points[point_index2]:
                points += _dfs_point(point_index2, point_lines, line_points, check_points)

    return points + [point_index]


x, y, g = get_gaussian_filter(sigma)
dx, dy = np.gradient(g)
ux, vx = separate_matrix(dx, 'x')
uy, vy = separate_matrix(dy, 'y')

bw = cv2.imread('books.jpg', 0)
img = cv2.imread('books.jpg')

ver = convolve2d(bw, dx, 'valid')

hor = convolve2d(bw, dy, 'valid')

grad_mag = calculate_grad_mag(ver, hor)

threshold = np.mean(grad_mag) * 3
new_grad = np.zeros(grad_mag.shape)
new_grad[grad_mag > threshold] = 255
edges = new_grad.astype('uint8')

lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=150, maxLineGap=6)

new_img = img.copy()
for item in lines:
    l = item[0]
    cv2.line(new_img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1)
cv2.imshow(RESULT_PATH + 'Q2_lines.jpg', new_img)

uniques = _unique_lines(lines, img.copy())

all_points, line_points, point_lines = _get_graph(uniques, lines)

check_points = [False for i in range(len(all_points))]
clusters = []
for i in range(len(all_points)):
    if check_points[i]:
        continue
    clusters.append(tuple(_dfs_point(i, point_lines, line_points, check_points)))

cnt = 1
for cluster in clusters:
    res = 'book %d corners are:' % cnt
    for item in cluster:
        res += '(%s, %s),  ' % (all_points[item][0], all_points[item][1])
    print (res)
    cnt += 1
