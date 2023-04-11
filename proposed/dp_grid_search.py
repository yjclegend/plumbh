from imagedata.realimage.samples import RealImage
import matplotlib.pyplot as plt

import numpy as np

def estimate_with_cod_guess():
    path = "data/testcase/cod_estimate/cod_estimate_1.bmp"
    im = RealImage(path)
    im.smoothPoly()
    cx, cy = 2011, 1244
    im.set_cod((cx, cy))
    im.build_system_no_decenter()

def estimate_with_cod_range():
    path = "data/testcase/cod_estimate/cod_estimate_1.bmp"
    im = RealImage(path)
    im.smoothSpline()
    cx, cy = 2011, 1244
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    z = list()
    search = np.arange(-10, 10)

    x, y = np.meshgrid(search, search)
    grid = np.column_stack([x.flatten(), y.flatten()])
    minz = 99999
    minidx = 0
    for i in range(grid.shape[0]):
        cod = grid[i, 0] + cx, grid[i, 1] + cy
        im.set_cod(cod)
        im.cal_norm()
        im.build_system_no_decenter()
        residual = im.cal_residual()
        if residual < minz:
            minz = residual
            minidx = i
        z.append(residual)
    print(grid[minidx])
    z = np.array(z)
    z = np.reshape(z, (z.shape[0], 1))
    ax.plot_surface(grid[:, 0], grid[:, 1], z)
    plt.show()
    
    

    # plt.imshow(im.image)
    # for seg in im.segments:
    #     plt.scatter(seg.segment[:, 0], seg.segment[:, 1])
    #     plt.scatter(seg.restored[:, 0], seg.restored[:, 1])
    # plt.show()
if __name__ == "__main__":
    
    estimate_with_cod_guess()