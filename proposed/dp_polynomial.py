import matplotlib.pyplot as plt

from imagedata.realimage.DataSet import DataSet
def test1():
    paths = ["data/testcase/cod_estimate/cod_estimate_1.bmp"]
    cx, cy = 2008, 1315

    ds = DataSet(paths)
    ds.set_cod((cx, cy))
    ds.estimate_poly(degree=2)

    res = ds.cal_residual()
    print("min res:", res)
    plt.imshow(ds.images[0].image)
    for seg in ds.segments:
        plt.scatter(seg.sample[:, 0], seg.sample[:, 1], color='blue')
        plt.scatter(seg.restored[:, 0], seg.restored[:, 1], color='red')
    plt.show()

if __name__ == "__main__":
    test1()