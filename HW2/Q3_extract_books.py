import cv2
import math
import numpy as np

books = [
    [(809, 960),  (618, 663),  (407, 785),  (601, 1096)],
    [(146, 698),  (200, 415),  (403, 455),  (344, 736)],
    [(369, 99),  (652, 207),  (584, 401),  (303, 293)]
]


def _get_dis(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def _get_h_and_w(point1, point2, point3, point4):
    height = np.mean([_get_dis(point1, point2), _get_dis(point3, point4)])
    width = np.mean([_get_dis(point2, point3), _get_dis(point4, point1)])

    if width > height:
        tmp = height
        height = width
        width = tmp

    return int(height)+1, int(width)+1


def _get_matrix_transfrom(point1, point2, point3, pointp1, pointp2, pointp3):
    # acos x1 + -asin y1 + tx = xp1
    a = np.array([[point1[0], point1[1], 1], [point2[0], point2[1], 1], [point3[0], point3[1], 1]])
    b = np.array([pointp1[0], pointp2[0], pointp3[0]])
    cos, sin, tx = np.linalg.solve(a, b)

    # asin x1 + acos y1 + ty = yp1
    a = np.array([[point1[0], point1[1], 1], [point2[0], point2[1], 1], [point3[0], point3[1], 1]])
    b = np.array([pointp1[1], pointp2[1], pointp3[1]])
    sin1, cos1, ty = np.linalg.solve(a, b)

    return np.array([[cos, sin, tx], [sin1, cos1, ty]])


if __name__ == "__main__":
    img = cv2.imread('books.jpg')

    for i in range(len(books)):
        book = books[i]

        height, width = _get_h_and_w(*book)

        bookp = [(0, 0), (0, height), (width, height), (width, 0)]

        M = _get_matrix_transfrom(book[0], book[1], book[2], bookp[0], bookp[1], bookp[2])

        affined_img = cv2.warpAffine(img, M, (width, height))

        cv2.imwrite('Q3_book%d.jpg' % (i+1), affined_img)






