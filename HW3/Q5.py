import matplotlib.pyplot as plt
import cv2

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

img = img_as_float(cv2.imread('im023.jpg'))
segments_slic2 = slic(img, n_segments=500, compactness=1, sigma=20)


fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
fig.tight_layout()

ax.imshow(mark_boundaries(img, segments_slic2))
ax.set_title("SLIC's method")

ax.set_xticks(())
ax.set_yticks(())
fig.savefig('im09.jpg')
