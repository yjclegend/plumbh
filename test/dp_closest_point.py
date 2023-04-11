import numpy as np
import matplotlib.pyplot as plt

from imagedata.realimage.PolygonImage import PolygonImage


path = 'data/testcase/param_estimate1/Image_20220816102411064.bmp'
pi = PolygonImage(path)
seg = pi.segments[0]
cod = np.array([1940, 1300]).reshape((1, 2))
plt.imshow(pi.image)
plt.scatter(seg.sample[:, 0], seg.sample[:, 1], s=16)

plt.show()

seg.set_cod(1940, 1300)