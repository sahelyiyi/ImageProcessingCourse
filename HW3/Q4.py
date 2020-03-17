import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import felzenszwalb as felzenszwalb_method
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

img = img_as_float(cv2.imread('im023.jpg'))
felzenszwalb = felzenszwalb_method(img, scale=100, sigma=2.5, min_size=50)


fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
fig.tight_layout()

ax.imshow(mark_boundaries(img, felzenszwalb))
ax.set_title("Felzenszwalbs's method")

ax.set_xticks(())
ax.set_yticks(())
fig.savefig('im08.jpg')
