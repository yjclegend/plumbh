from proposed.model.samples import PlumbLine
from proposed.model.camera_model import CameraModel

from imagedata.synthetic.gendata import genLines, genNormLine
from imagedata.synthetic.imaging import distort


import numpy as np
import matplotlib.pyplot as plt

def test_line_norm():
    scale = 2000
    k = -0.1
    axis = 0
    tilt = 0.0
    alpha = 1.000
    gamma = 0.000
    norm_rad = 0
    line = genNormLine(norm_rad, 0.5, crop=False, r=1000)

    f=2000
    distorted = distort(line, -0.1, axis, tilt, alpha, gamma, f)
    cm = CameraModel(f, (0, 0))
    line1 = PlumbLine(distorted, cm)

    line1.update_model()
    line1.neibour_line_fit()

    error = abs(norm_rad - line1.norm_rad)
    print("rad_error: ", error)

def fix_error(error):
    if error > np.pi/2:
        error -= np.pi
    if error > np.pi/2:
        error -= np.pi
    return abs(error)

def test2():
    scale = 2000
    k = -0.1
    axis = 0
    tilt = 0.08
    alpha = 1.0005
    gamma = 0.0005
    lines = genLines(24, crop=True)
    rads = np.arange(24) * 2 * np.pi / 24
    cm = CameraModel(scale, (0, 0), 3)
    line_fit = []
    err_list = []
    for i in range(len(lines)):
        line = lines[i]
        dist = distort(line, k, axis, tilt, alpha, gamma, scale)
        pl = PlumbLine(dist, cm)
        pl.update_model()
        pl.neibour_line_fit()
        error = abs(rads[i] - pl.norm_rad)
        error2 = abs(rads[i] - pl.line_rad)
        err_list.append(fix_error(error))
        line_fit.append(fix_error(error2))
    plt.scatter(rads, line_fit, label='line fit')
    plt.scatter(rads, err_list, label='NVI')
    plt.xlabel('ground truth angle(rad)')
    plt.ylabel('normal vecotor angle error(rad)')
    plt.legend()
    plt.show()
    print(err_list)

def test3():
    scale = 2000
    k = -0.1
    axis = 0
    tilt = 0.1
    alpha = 1.0001
    gamma = 0.0001
    line = genNormLine(0, 1, False, 1000)
    dist = distort(line, k, axis, tilt, alpha, gamma, scale, discrete=False)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    norm_list = []
    for i in range(-100, 100, 5):
        for j in range(-100, 100, 5):
            cm = CameraModel(scale, (i, j), 3)
            pl = PlumbLine(dist, cm)
            pl.update_model()
            pl.neibour_line_fit()
            norm_list.append([i, j, pl.norm_rad])
    norms = np.array(norm_list)
    ax.scatter(norms[:, 0], norms[:, 1], norms[:, 2])
    norm = np.arange(-100, 100)
    ax.plot(norm, np.zeros_like(norm), 0, color='green', label='norm line')
    # ax.set_zlabel('norm angle estimation')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # test_line_norm()
    # test2()
    test3()