import numpy as np
import matplotlib.pyplot as plt
from common.evaluation import straightness

from common.zebra import prepare_lines
from proposed.evaluation import calibrate
from proposed.model.camera_model import CameraModel
from proposed.model.samples import PlumbLine


def test1(name):
    lines = prepare_lines('data/'+ name)
    # for line_set in lines:
    #     for line in line_set:
    #         plt.scatter(line[:, 0], line[:, 1])
    # plt.show()
    # exit()
    pls:list[PlumbLine] = []
    train_pls:list[PlumbLine] = []
    test_pls:list[PlumbLine] = []

    scale = 2000
    cm = CameraModel(scale, (1940, 1374), 3, homo=True, lr=1/len(lines))
    
    set_num = 0
    for line_set in lines:
        set_num += 1
        if set_num % 3 != 1:
            continue
        num = len(line_set)
        quater = num //4
        for i in range(num):
            pl = PlumbLine(line_set[i], cm)
            if i%quater ==0:# < 1 or i > num - 2 or i == num//2:
                train_pls.append(pl)
            else:
                test_pls.append(pl)
    pls = train_pls + test_pls
    pls.pop(28)
    # train_pls = pls[10:11]
    print(len(pls))
    # cm.lr = 1 / len(train_pls)
    # for pl in train_pls:
    #     plt.scatter(pl.origin[:, 0], pl.origin[:, 1])
    # plt.show()
    # exit()

    # train_pls = train_pls[:1]
    cm.estimate(train_pls, iters=200)
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
    for lin in train_pls:
        res_list.append(lin.avg_res*scale**2)
    plt.plot(res_list)
    plt.show()
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


def case1(suffix):
    name = "zebra202303220900"
    lines = prepare_lines('data/'+ name)

    pls:list[PlumbLine] = []
    train_pls:list[PlumbLine] = []
    set_num = 0

    cm = CameraModel(2000, (1940, 1374), 3, homo=True)
    for line_set in lines:
        set_num += 1
        if set_num % 3 != 1:
            continue
        num = len(line_set)
        quater = num //4
        for i in range(num):
            pl = PlumbLine(line_set[i], cm)
            if i%quater ==0:# < 1 or i > num - 2 or i == num//2:
                train_pls.append(pl)
            pls.append(pl)
    cm.hlr = 1/len(train_pls)
    calibrate(cm, train_pls, pls)
    cm.save_param(name + '_' + suffix)


if __name__ == "__main__":
    # calibrate("zebra202303210835")

    # cm = CameraModel(2000, (1940, 1374), 3, homo=True, lr=1/len(lines))
    # test1("zebra202303220900")
    case1("1")