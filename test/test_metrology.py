import cv2

import matplotlib.pyplot as plt

from proposed.dp_StraightLine import StraightLineCOD


def __test1():
    sl = StraightLineCOD()
    sl.find_segments("data/testcase/param_estimate1")
    # for seg in sl.segments:
    #     plt.scatter(seg.segment[:, 0], seg.segment[:, 1])
    # plt.show()
    sl.calibrate(sr = 1, ref_plain="data/testcase/homography/homo1.bmp")
    corners = sl.find_chessboard("data/testcase/homography/homo1.bmp")
    corners_restore = sl.undistort_points(corners)
    corners_rectify = sl.rectify(corners_restore)
    image = cv2.imread("data/testcase/homography/homo1.bmp")
    plt.imshow(image)
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.scatter(corners_restore[:, 0], corners_restore[:, 1])
    plt.scatter(corners_rectify[:, 0], corners_rectify[:, 1])
    plt.show()


__test1()