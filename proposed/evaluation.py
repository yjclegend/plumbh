from common.evaluation import straightness
from proposed.model.camera_model import CameraModel
from proposed.model.samples import PlumbLine

import numpy as np
import matplotlib.pyplot as plt
def calibrate(cm:CameraModel, lines:list[PlumbLine], test_pls:list[PlumbLine]):
    cm.estimate(lines, iters=200)
    # cm.save_param("zebra202303220900")
    cod_list = np.array(cm.cod_list)

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(cod_list[:, 0])
    axs[1].plot(cod_list[:, 1])
    axs[2].plot(cm.residual)
    plt.show()
    print(cm.cod, cm.dist_coeff)
    print(cm.homography)
    # training error
    res_list = list()
    for lin in lines:
        res_list.append(lin.avg_res*cm.scale**2)
    plt.plot(res_list)
    plt.show()
    print("mean residual: ", np.mean(np.array(res_list)))

    # test error
    test_res = list()
    for lin in test_pls:
        lin.update_model()
        unplumb = lin.restore_image()
        plt.scatter(lin.origin[:, 0], lin.origin[:, 1], color='blue')
        plt.scatter(unplumb[:, 0], unplumb[:, 1], color='red')
        # test_res.append(lin.straightness())
        test_res.append(straightness(unplumb))
    plt.show()
    print("test residual: ", np.mean(np.array(test_res)))