from common.evaluation import straightness
from common.zebra import prepare_lines
from proposed.model.camera_model import CameraModel
from proposed.model.samples import PlumbLine

import numpy as np
import matplotlib.pyplot as plt

def given_COD(cod, homo=None):
    lines = prepare_lines('data/zebra202303210835/')

    scale = 2000
    pls:list[PlumbLine] = []
    train_pls:list[PlumbLine] = []
    test_pls:list[PlumbLine] = []
    train_ratio = 1

    cm = CameraModel(scale, cod, 5, homo=False, lr=0.1/len(lines))
    if homo is not None: 
        cm.homography = homo
    for line_set in lines:
        num = len(line_set)
        sp = int(train_ratio * num)
        np.random.shuffle(line_set)
        for i in range(num):
            pl = PlumbLine(line_set[i], cm)
            if i < sp:
                train_pls.append(pl)
            else:
                test_pls.append(pl)

    pls = train_pls + test_pls


    cm.estimate(train_pls, iters=1)
    cm.save_param("zebra202303210835")
    print(cm.dist_coeff)
    cm.cod = cod

    # training error
    res_list = list()
    for lin in train_pls:
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


# given_COD((2000, 1220))
# given_COD((1920, 1374))

# given_COD((1980, 1250))

# given_COD((1953, 1392))

homo = np.array([[ 1.00006976e+00,  3.48715882e-05,  0.00000000e+00],
 [ 9.04076167e-05,  9.99891419e-01,  0.00000000e+00],
 [-1.27657440e-04, -7.95139821e-04,  1.00000000e+00]])
given_COD((1962, 1244), homo)