import cv2
import numpy as np
from PIL import Image

IMAGE_PATH = 'results/'

img_path = '01.png'
img = cv2.imread(img_path)

old_img = cv2.imread('01.png')
old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
old_points = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=150, qualityLevel=0.05, minDistance=5, blockSize=5)

new_img = cv2.imread('02.png')
new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
next_points, points_status, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, winSize=(15, 15),
                                                         maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

mask = np.zeros_like(old_img)
for new, old in zip(next_points[points_status == 1], old_points[points_status == 1]):
    old_y, old_x = int(old[0]), int(old[1])
    new_y, new_x = int(new[0]), int(new[1])
    mask = cv2.line(mask, (old_y, old_x), (new_y, new_x), [0, 255, 255], 3)
    old_img = cv2.circle(old_img, (new_y, new_x), 3, [0, 255, 255], -1)
    new_img = cv2.circle(new_img, (new_y, new_x), 3, [0, 255, 255], -1)

result = cv2.add(old_img, mask)

cv2.imwrite(IMAGE_PATH + 'im5.jpg', result)

flow_img0 = Image.fromarray(result)
flow_img1 = Image.fromarray(new_img)
flow_img0.save(IMAGE_PATH + 'flow.gif', format='GIF', append_images=[flow_img1], save_all=True, duration=500, loop=0)
