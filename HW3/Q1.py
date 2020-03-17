import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift
from collections import defaultdict
import numpy as np


def draw_points(points):
    x = [item[0] for item in points]
    y = [item[1] for item in points]

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(x, y)
    fig.savefig('im01.jpg')


def get_clusters(points, pred_y):
    clusters = defaultdict(list)
    for i in range(len(pred_y)):
        clusters[pred_y[i]].append(points[i])

    for item in clusters:
        clusters[item] = np.array(clusters[item])
    return clusters


def k_means(points, num_tries=2):
    fig, axes = plt.subplots(num_tries, num_tries)
    for num_try1 in range(num_tries):
        for num_try2 in range(num_tries):
            kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300)
            pred_y = kmeans.fit_predict(points)
            clusters = get_clusters(points, pred_y)

            axes[num_try1, num_try2].scatter(clusters[0][:, 0], clusters[0][:, 1], color='red')
            axes[num_try1, num_try2].scatter(clusters[1][:, 0], clusters[1][:, 1], color='blue')
    fig.savefig('im02.jpg')


def k_means_changed(points, changed_points):
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300)
    pred_y = kmeans.fit_predict(changed_points)
    clusters = get_clusters(points, pred_y)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(clusters[0][:, 0], clusters[0][:, 1], color='red')
    ax.scatter(clusters[1][:, 0], clusters[1][:, 1], color='blue')
    fig.savefig('im04.jpg')


def mean_shift(points):
    clustering = MeanShift().fit(points)
    pred_y = clustering.predict(points)
    clusters = get_clusters(points, pred_y)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(clusters[0][:, 0], clusters[0][:, 1], color='red')
    ax.scatter(clusters[1][:, 0], clusters[1][:, 1], color='blue')
    fig.savefig('im03.jpg')


def change_space(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


with open('Points.txt', 'r') as f:
    data = f.read()

data = data.split('\n')
pints_cnt, points_data = data[0], data[1:-1]


points = []
for item in points_data:
    x, y = item.split(' ')
    points.append((float(x), float(y)))

k_means(points)
mean_shift(points)

changed_points = []
for point in points:
    changed_points.append(change_space(*point))

k_means_changed(points, changed_points)
