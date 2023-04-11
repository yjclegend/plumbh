import numpy as np
import matplotlib.pyplot as plt
from common.evaluation import straightness
from common.sceneline import scene_lines

from proposed.model.camera_model import CameraModel
from proposed.model.samples import PlumbLine


def calibrate(name):
    lines = scene_lines('data/'+ name)
    # for line_set in lines:
    #     for line in line_set:
    #         plt.scatter(line[:, 0], line[:, 1])
    # plt.show()
    # exit()
    pls:list[PlumbLine] = []

    scale = 2000
    cm = CameraModel(scale, (1940, 1374), 3, homo=True, lr=1/len(lines))
    
    for line in lines:
        pl = PlumbLine(line, cm)
        pls.append(pl)

    print(len(pls))
    # cm.lr = 1 / len(train_pls)
    # for pl in train_pls:
    #     plt.scatter(pl.origin[:, 0], pl.origin[:, 1])
    # plt.show()
    # exit()

    # train_pls = train_pls[:1]
    cm.estimate(pls, iters=200)
    cm.save_param(name + '_full')
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
    for lin in pls:
        res_list.append(lin.avg_res*scale**2)
    print("mean residual: ", np.mean(np.array(res_list)))

    # test error
    test_res = list()
    for lin in pls:
        lin.update_model()
        unplumb = lin.restore_image()
        plt.scatter(lin.origin[:, 0], lin.origin[:, 1], color='blue')
        plt.scatter(unplumb[:, 0], unplumb[:, 1], color='red')
        # test_res.append(lin.straightness())
        test_res.append(straightness(unplumb))
    plt.show()
    print("test residual: ", np.mean(np.array(test_res)))


if __name__ == "__main__":

    calibrate("scene202303221500")