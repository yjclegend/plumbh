from common.evaluation import straightness
from common.zebra import prepare_lines
from reference.zhang import Zhang

import numpy as np
import matplotlib.pyplot as plt

zc = Zhang()
zc.load_param('chess202303211245')
# zc.load_param('chess202303211245_k1k2k3')
zhang_res = []

lines = prepare_lines('data/zebra202303220900/')
ori_res = []
line_samples = list()
for line_set in lines:
    for line in line_set:
        ori_res.append(straightness(line))
        # zhang
        unzhang = zc.undistort_points(line)
        zhang_res.append(straightness(unzhang))
        line_samples.append(line)

diff_res:np.ndarray = np.abs(np.array(ori_res) - np.array(zhang_res))

orders = np.argsort(ori_res)

for i in range(20):
    plt.scatter(line_samples[orders[i]][:, 0], line_samples[orders[i]][:, 1])
plt.show()