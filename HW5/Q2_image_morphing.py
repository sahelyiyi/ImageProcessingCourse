import cv2
import numpy as np
import dlib
import random
from scipy.spatial import Delaunay
from subprocess import Popen, PIPE
from PIL import Image, ImageDraw


FRAMES_NUM = 1000
RESULT_PATH = 'results/'
RESULT_VIDEO = 'morphing.mp4'


def _change_coordinates(triangle, x, y):
    points = np.array([[triangle[0][0] - x, triangle[0][1] - y],
                       [triangle[1][0] - x, triangle[1][1] - y],
                       [triangle[2][0] - x, triangle[2][1] - y]], np.int32)
    return points


def _get_cropped_triangle(triangle, img, current_triangle, current_w, current_h):
    rect = cv2.boundingRect(np.array([triangle]))
    (x, y, w, h) = rect
    cropped_triangle = img[y: y + h, x: x + w].astype('float32')

    points = _change_coordinates(triangle, x, y)
    current_points = _change_coordinates(current_triangle, x, y)

    affine = cv2.getAffineTransform(points.astype('float32'), current_points.astype('float32'))
    warped_triangle = cv2.warpAffine(cropped_triangle, affine, (current_w, current_h), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return warped_triangle


def morph(triangle1, img1, triangle2, img2, current_triangle, current_image, alpha):
    rect = cv2.boundingRect(np.array([current_triangle]))
    (x, y, w, h) = rect

    wraped_triangle1 = _get_cropped_triangle(triangle1, img1, current_triangle, w, h)
    wraped_triangle2 = _get_cropped_triangle(triangle2, img2, current_triangle, w, h)

    cropped_mask = np.zeros((h, w, 3), np.int32)
    points = _change_coordinates(current_triangle, x, y)
    cv2.fillConvexPoly(cropped_mask, points, (1, 1, 1))

    aggregate_triangles = ((1.0 - alpha) * wraped_triangle1 + alpha * wraped_triangle2).astype(img1.dtype)

    current_image[y: y + h, x: x + w] = current_image[y: y + h, x: x + w] * (1 - cropped_mask) + aggregate_triangles * cropped_mask


def get_facial_points(img, detector, predictor):
    points = []

    faces = detector(img)
    for face in faces:
        landmarks = predictor(img, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))

    corners_y = [1, img.shape[1]-1]
    corners_x = [1, img.shape[0]-1]
    for corner_y in corners_y:
        for corner_x in corners_x:
            points.append((corner_y, corner_x))
            if corner_y > 1:
                points.append((corner_y//2, corner_x))
            if corner_x > 1:
                points.append((corner_y, corner_x//2))
            if corner_x > 1 and corner_y > 1:
                points.append((corner_y//2, corner_x//2))

    return points


def draw_triangles(in_img, out_img, points, triangles, colors=None):
    im = Image.open(in_img)
    draw = ImageDraw.Draw(im)
    cnt = 0
    for item in triangles:
        if colors:
            draw.polygon([points[item[0]], points[item[1]], points[item[2]]], fill=colors[cnt])
        else:
            draw.polygon([points[item[0]], points[item[1]], points[item[2]]])
        cnt += 1
    im.save(RESULT_PATH + out_img)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image1 = cv2.imread('morphing_source.jpg', 1)
points1 = get_facial_points(image1, detector, predictor)
triangles1 = Delaunay(points1).simplices

image2 = cv2.imread('morphing_target.jpg', 1)
points2 = get_facial_points(image2, detector, predictor)
triangles2 = Delaunay(points2).simplices

triangles = []
for triangle in triangles1:
    if triangle in triangles2:
        triangles.append(triangle)

colors = []
for i in range(len(triangles)):
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    r = random.randint(0, 256)
    colors.append((b, g, r))

draw_triangles('morphing_source.jpg', 'morphing_source_triangle.jpg', points1, triangles)
draw_triangles('morphing_target.jpg', 'morphing_target_triangle.jpg', points2, triangles)

draw_triangles('morphing_source.jpg', 'morphing_source_filled_triangle.jpg', points1, triangles, colors)
draw_triangles('morphing_target.jpg', 'morphing_target_filled_triangle.jpg', points2, triangles, colors)

p = Popen(
    ['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(FRAMES_NUM), '-s', str(image1.shape[1]) + 'x' + str(image1.shape[0]), '-i', '-',
     '-c:v', 'libx264', '-crf', '25', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', RESULT_PATH + RESULT_VIDEO],
    stdin=PIPE)

for k in range(FRAMES_NUM):
    current_points = []
    alpha = k * 1.0 / (FRAMES_NUM - 1)
    for i in range(len(points1)):
        x = int((1 - alpha) * points1[i][0] + alpha * points2[i][0])
        y = int((1 - alpha) * points1[i][1] + alpha * points2[i][1])
        current_points.append((x, y))

    current_image = np.zeros(image1.shape, dtype=image1.dtype)

    for item in triangles:
        x, y, z = item[0], item[1], item[2]

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [current_points[x], current_points[y], current_points[z]]

        morph(t1, image1, t2, image2, t, current_image, alpha)

    temp_res = cv2.cvtColor(np.uint8(current_image), cv2.COLOR_BGR2RGB)
    res = Image.fromarray(temp_res)
    res.save(p.stdin, 'JPEG')

p.stdin.close()
p.wait()
