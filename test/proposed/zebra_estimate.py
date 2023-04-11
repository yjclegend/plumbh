from common.zebra import load_lines
from proposed.model.camera_model import CameraModel

import numpy as np
import matplotlib.pyplot as plt

from proposed.model.samples import PlumbLine

lines = load_lines('zebra202303220900', smooth=True)

flatten_lines = list()
for lineset in lines:
    for line in lineset:
        if len(line) > 3000:
            flatten_lines.append(line)




scale = 2000
for i in range(8):
    np.random.shuffle(flatten_lines)
    train_lines = flatten_lines[:40]
    cm = CameraModel(scale, (1940, 1374), 3, homo=True, clr = 200/ len(train_lines), hlr=1/len(train_lines))
    train_pls = list()
    
    for line in train_lines:
        pl = PlumbLine(line, cm)
        train_pls.append(pl)

    cm.estimate(train_pls, iters=400)
    cm.save_param("zebra202303220900_" + str(i))

cod_list = np.array(cm.cod_list)

fig, axs = plt.subplots(3, 1)
axs[0].plot(cod_list[:, 0])
axs[1].plot(cod_list[:, 1])
axs[2].plot(cm.residual)
plt.show()
print(cm.cod, cm.dist_coeff)
print(cm.homography)


