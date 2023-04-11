from common.evaluation import straightness
from common.zebra import load_lines


import matplotlib.pyplot as plt
import numpy as np
from proposed.model.camera_model import CameraModel
from proposed.model.samples import PlumbLine

from reference.zhang import Zhang

def straightness_test(dataset, param):
    lines = load_lines(dataset, smooth=True)
    # lines = load_lines('zebra202303220900', smooth=True)
    ori_res = []

    zc = Zhang()
    zc.load_param('chess202303211245')
    # zc.load_param('chess202303211245_k1k2k3')
    zhang_res = []


    cm = CameraModel(2000, (0, 0), 3)
    # cm.load_param('zebra202303210835_1')
    # cm.load_param('zebra202303210835')
    # cm.load_param('zebra202303220900_full')
    cm.load_param(param)
    # cm.load_param('zebra202304030915_202304031646')

    # cm.load_param('scene202303221500_full')
    print(cm.homography)
    plumb_res = []
    idx = 0
    for line_set in lines:
        for line in line_set:
            idx += 1
            ori_res.append(straightness(line))
            # zhang
            unzhang = zc.undistort_points(line.astype('float64'))
            zhang_res.append(straightness(unzhang))

            # proposed
            # unplumb = cm.undistort_points(line)
            pl = PlumbLine(line, cm)
            pl.update_model()
            unplumb = pl.restore_image()
            plumb_res.append(straightness(unplumb))
            if idx > 0 and idx < 10:
                plt.scatter(line[:, 0], line[:, 1], color='blue', label='origin')
                plt.scatter(unzhang[:, 0], unzhang[:, 1], color='red',label='zhang')
                plt.scatter(unplumb[:, 0], unplumb[:, 1], color='green', label='plumb')
    print('total lines: ', idx)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    ori_res = np.array(ori_res)
    zhang_res = np.array(zhang_res)
    plumb_res = np.array(plumb_res)
    idx = np.where(zhang_res < 1)
    ori_res = ori_res[idx]
    zhang_res = zhang_res[idx]
    plumb_res = plumb_res[idx]
    plt.plot(ori_res, label='origin')
    plt.plot(zhang_res, label='zhang')
    plt.plot(plumb_res, label='plumb')
    plt.legend()
    plt.show()
    print("mean residual ori: ", np.mean(ori_res))
    print("mean residual zhang: ", np.mean(zhang_res))
    print("mean residual plumb: ", np.mean(plumb_res))


if __name__=="__main__":
    # for i in range(8):
    #     # straightness_test('zebra202304030915', "zebra202304030915_" + str(i))
    #     straightness_test('zebra202304030915', "zebra202303220900_" + str(i))
    straightness_test('zebra202304030915', "scene202303221500_full")
