
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')

from imagedata.synthetic.gendata import genLines, genNormLine
from imagedata.synthetic.imaging import distort
from proposed.model.samples import PlumbLine
def estimate_distortion(lines:list[PlumbLine], cod, degree=2):
    line_num = len(lines)
    x_list = list()
    y_list = list()
    for i in range(line_num):
        line = lines[i]
        line.update_model(cod)
        x, y = line.line_equation(degree)
        sparse = np.zeros((x.shape[0], line_num))
        sparse[:, i] = 1
        sx = np.column_stack([x, sparse])
        sy = y[:, np.newaxis]
        x_list.append(sx)
        y_list.append(sy)
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    params = np.linalg.lstsq(x, y)[0]
    cs = params[-line_num:]
    distortion = params[:-line_num]
    # print(cs)
    # print(distortion)
    sdx, sdy = 0, 0
    for i in range(line_num):
        line = lines[i]
        line.restore(cs[i, 0], distortion)
        # print('mean squared error: ', line.avg_res * 2000**2)
        dx, dy = line.gradient()
        sdx += dx
        sdy += dy
    return sdx, sdy, cs, distortion

def iter_estimate(lines:list[PlumbLine], cod):
    cx, cy = cod
    cod_list = [cod]
    dist_list = []
    for i in range(100):
        sdx, sdy, cs, dist_coeff = estimate_distortion(lines, (cx, cy), 3)
        cx += sdx
        cy += sdy
        cod_list.append((cx, cy))
        dist_list.append(dist_coeff)
    cod_list = np.array(cod_list)
    dist_list = np.array(dist_list)
    # plt.plot(cod_list[:, 0])
    # plt.plot(cod_list[:, 1])
    # plt.show()
    return cod_list[-1], dist_list[-1], cs
    
    # plt.plot(dist_list[:, 0])
    # plt.plot(dist_list[:, 1])
    # plt.show()

def test2():
    lines = genLines(12)
    lines.extend(genLines(12, 0.5))
    pls:list[PlumbLine] = list()
    scale = 2000
    k = -0.01
    axis = 90
    tilt = 0.08
    alpha = 1.000
    gamma = 0.000
    for line in lines:
        dist = distort(line, k, axis, tilt, alpha, gamma, scale)
        pls.append(PlumbLine(dist, scale))
    cod, dist_coeff, _ = iter_estimate(pls[:], (0, 0))
    print(cod, dist_coeff)
    # res_list = list()
    # for lin in pls:
    #     res_list.append(lin.avg_res)
    # plt.scatter(np.arange(24) * 2 * np.pi / 24, res_list)
    # plt.show()
    # test_lines = genLines(20, 0.8)
    # st_list = list()
    # for lin in test_lines:
    #     dist = distort(lin, k, axis, tilt, alpha, gamma, scale)
    #     pl = PlumbLine(dist, scale)
    #     unline = pl.undistort(cod, dist_coeff)
    #     print("straightness: ", pl.straightness())
    #     st_list.append(pl.straightness())
    #     plt.scatter(lin[:, 0], lin[:, 1], color='blue')
    #     plt.scatter(unline[:, 0], unline[:, 1], color='red')
    # plt.show()
    # plt.scatter(np.arange(20) * 2 * np.pi / 20, st_list)
    # plt.show()

if __name__ == "__main__":
    test2()