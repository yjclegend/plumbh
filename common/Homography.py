import numpy as np
import cv2, math

def homography(pers, rect):
    assert(pers.shape[0] == rect.shape[0])
    equations = list()
    for i in range(pers.shape[0]):
        equation1 = [-pers[i, 0], -pers[i, 1], -1, 0, 0, 0, rect[i, 0]*pers[i, 0], rect[i, 0]*pers[i, 1], rect[i, 0]]
        equation2 = [0, 0, 0, -pers[i, 0], -pers[i, 1], -1, rect[i, 1]*pers[i, 0], rect[i, 1]*pers[i, 1], rect[i, 1]]
        equations.append(equation1)
        equations.append(equation2)
    equations = np.array(equations)
    A = equations[:, :-1]
    b = -1 * equations[:, -1]

    h = np.linalg.lstsq(A, b)[0]
    h = np.append(h, 1)
    h = np.reshape(h, (3, 3))
    return h

def homography_svd(pers, rect):
    assert(pers.shape[0] == rect.shape[0])
    equations = list()
    for i in range(pers.shape[0]):
        equation1 = [-pers[i, 0], -pers[i, 1], -1, 0, 0, 0, rect[i, 0]*pers[i, 0], rect[i, 0]*pers[i, 1], rect[i, 0]]
        equation2 = [0, 0, 0, -pers[i, 0], -pers[i, 1], -1, rect[i, 1]*pers[i, 0], rect[i, 1]*pers[i, 1], rect[i, 1]]
        equations.append(equation1)
        equations.append(equation2)

    equations = np.array(equations)
    print(equations.shape)
    U, singular, V_transpose = np.linalg.svd(equations)

    h = np.reshape(V_transpose[-1], (3, 3))
    return h

def rectify(points, h):
    points_homo = np.column_stack([points, np.ones((points.shape[0], 1))])
    rectify = np.dot(h, points_homo.T).T
    rectify[:, 0] /= rectify[:, 2]
    rectify[:, 1] /= rectify[:, 2]
    return rectify[:, :2]

def fundamental(pd, pc):
    c1 = pd[:, 0] * pc[:, 0]
    c2 = pd[:, 0] * pc[:, 1]
    c3 = pd[:, 0]
    c4 = pc[:, 0] * pd[:, 1]
    c5 = pd[:, 1] * pc[:, 1]
    c6 = pd[:, 1]
    c7 = pc[:, 0]
    c8 = pc[:, 1]
    c9 = np.ones_like(c8)
    equations = np.column_stack([c1, c2, c3, c4, c5, c6, c7, c8, c9])
    U, singular, V_transpose = np.linalg.svd(equations)

    F = np.reshape(V_transpose[-1], (3, 3))
    return F

def left_epipole(f):
    U, S, V = np.linalg.svd(f)
    e = V[-1]
    return e / e[2]



### test
from calibration.Calibration import Calibration

import matplotlib.pyplot as plt
from common.Homography import fundamental, homography, left_epipole
def test1():
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    # objp = ca.objp * 100 + 1000
    ca.homography_svd(corners, ca.objp)
    rectify = ca.rectify(corners)
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.scatter(ca.objp[:, 0], ca.objp[:, 1])
    plt.scatter(rectify[:, 0], rectify[:, 1])
    plt.show()


def test2():
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    objp = ca.objp
    objp[:, 0] = (objp[:, 0] - 1000)*2
    objp[:, 1] = (objp[:, 1] - 1000)
    h = ca.homography(corners, ca.objp)
    print(h)
    # print(h[2, 0] / h[2, 1])
    rectify = ca.rectify(corners)
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.scatter(objp[:, 0], objp[:, 1])
    plt.scatter(rectify[:, 0], rectify[:, 1])
    plt.show()

def test3():
    import numpy as np
    objp = np.mgrid[0:11, 0:8].T.reshape((88, 2))
    objp = objp 
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    h = homography(objp, corners)
    print(h)
    objp2 = objp
    objp2[:,  0] += 1000
    objp2 = objp2*2
    # objp2[:,  0] += 1000
    h = homography(objp, objp2)
    print(h)

def test4():
    ca = Calibration()
    corners = ca.find_chessboard("data/testcase/homography/homo1.bmp")
    objp = ca.objp
    f_matrix = fundamental(corners, objp)
    print(f_matrix)
    e = left_epipole(f_matrix.T)
    print(e)

if __name__ == "__main__":
    test1()