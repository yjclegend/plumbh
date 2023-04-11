import matplotlib.pyplot as plt
import time, os

from imagedata.realimage.DataSet import DataSet

def estimate_with_cod_guess(ds:DataSet, cod):
    cx, cy = cod
    ds.set_cod((cx, cy))
    ds.estimate_no_decenter()
    res = ds.cal_residual()
    print("min res:", res)
    # plt.imshow(ds.images[0].image)
    for seg in ds.segments:
        plt.scatter(seg.sample[:, 0], seg.sample[:, 1], color='blue', label='original')
        plt.scatter(seg.restored[:, 0], seg.restored[:, 1], color='red', label='undistort')
    # plt.legend()
    plt.show()

def estimate_grid(ds:DataSet, cod, sr=5):
    cx, cy = cod
    minres = 100000000
    best_i, best_j = 0, 0
    for i in range(-sr, sr):
        for j in range(-sr, sr):
            ds.set_cod((cx+i, cy+j))
            ds.estimate_no_decenter()
            res = ds.cal_residual()
            if res < minres:
                minres = res
                best_i, best_j = i, j
    print("min res:", minres)
    print(best_i, best_j)
    plt.imshow(ds.images[0].image)
    for seg in ds.segments:
        plt.scatter(seg.sample[:, 0], seg.sample[:, 1], color='blue', label='original')
        plt.scatter(seg.restored[:, 0], seg.restored[:, 1], color='red', label='undistort')
    # plt.legend()
    plt.show()
    


def test1():
    paths = ["data/testcase/cod_estimate/cod_estimate_1.bmp"]
    cx, cy = 2008, 1315

    ds = DataSet(paths)
    print("start ", time.time())
    ds.set_cod((cx, cy))
    print("1 ", time.time())
    ds.estimate_no_decenter(use_decenter=True)
    print("2 ", time.time())
    res = ds.cal_residual()
    print("3 ", time.time())
    print("min res:", res)
    plt.imshow(ds.images[0].image)
    for seg in ds.segments:
        plt.scatter(seg.sample[:, 0], seg.sample[:, 1], color='blue', label='original')
        plt.scatter(seg.restored[:, 0], seg.restored[:, 1], color='red', label='undistort')
    plt.legend()
    plt.show()



def test2():
    root = "data/testcase/param_estimate"
    files = os.listdir("data/testcase/param_estimate")
    for i in range(len(files)):
        files[i] = os.path.join(root, files[i])
    print(files)
    ds = DataSet(files)
    cod = 1940, 1300
    estimate_with_cod_guess(ds, cod)
    # estimate_grid(ds, cod)

if __name__ == "__main__":
    test2()
    # grid_search1() 