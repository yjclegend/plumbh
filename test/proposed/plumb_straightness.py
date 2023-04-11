from common.evaluation import straightness
from common.zebra import prepare_lines
from proposed.model.camera_model import CameraModel

import matplotlib.pyplot as plt
import numpy as np

ori_res = []
plumb_res = []
lines = prepare_lines('data/zebra202303210835/')
scale = 2000
cod = (0, 0)
cm = CameraModel(scale, cod, 9, homo=False, lr=1)
cm.load_param('chess202303211245_plumb')
for line_set in lines:
    for line in line_set:
        ori_res.append(straightness(line))
        unplumb = cm.undistort_points(line)
        plumb_res.append(straightness(unplumb))

        plt.scatter(line[:, 0], line[:, 1], color='blue')
        plt.scatter(unplumb[:, 0], unplumb[:, 1], color='red')
plt.show()


print("mean residual ori: ", np.mean(np.array(ori_res)))
print("mean residual zhang: ", np.mean(np.array(plumb_res)))

