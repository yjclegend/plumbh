import matplotlib.pyplot as plt

from proposed.dp_StraightLine import StraightLineCOD


def test1():
    sl = StraightLineCOD()
    sl.find_segments("data/testcase/param_estimate2")
    # for seg in sl.segments:
    #     plt.scatter(seg.segment[:, 0], seg.segment[:, 1])
    # plt.show()
    sl.calibrate((1950, 1310), sr=15)
    for seg in sl.segments:
        plt.scatter(seg.sample[:, 0], seg.sample[:, 1], color='blue')
        plt.scatter(seg.restored[:, 0], seg.restored[:, 1], color='red')
    plt.show()


def test2():
    sl = StraightLineCOD(use_k2=True, use_k3=True, use_decenter=False)
    sl.find_segments("data/testcase/param_estimate1")
    # sl.find_segments("data/testcase/param_estimate3")
    for seg in sl.segments:
        plt.scatter(seg.sample[:, 0], seg.sample[:, 1])
    plt.show()
    sl.calibrate((1975, 1295), sr=5)
    # sl.calibrate((310, 245), sr=5)
    # import cv2
    # image = cv2.imread('data/testcase/param_estimate1/Image_20220816102328865.bmp')
    # image = cv2.imread('data/testcase/param_estimate3/image_9.jpg')
    # plt.imshow(image)
    for seg in sl.segments:
        plt.scatter(seg.sample[:, 0], seg.sample[:, 1], color='blue')
        plt.scatter(seg.restored[:, 0], seg.restored[:, 1], color='red')
    plt.show()

test2()