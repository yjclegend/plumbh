import numpy as np
import cv2
import matplotlib.pyplot as plt
from common.Homography import homography, rectify
from common.zhang import calibration


from common.chessboard import findchessboard
from common.geometry import distancePoint


def rectify_residual(points, size=(11, 8)):
    left_top = points[0]
    right_top = points[size[0]-1]
    left_bottom = points[-size[0]]
    right_bottom = points[-1]
    top_edge = distancePoint(left_top, right_top)
    bottom_edge = distancePoint(left_bottom, right_bottom)
    left_edge = distancePoint(left_top, left_bottom)
    right_edge = distancePoint(right_top, right_bottom)
    print(top_edge, bottom_edge)
    print(left_edge, right_edge)

def no_calibration():
    objp = np.mgrid[0:11, 0:8].T.reshape((88, 2))
    
    corners = findchessboard("data/2023-02-10/1.bmp")
    h = homography(corners, objp)

    corners2 = findchessboard("data/2023-02-10/2.bmp")
    if corners2 is not None:
        rect = rectify(corners2, h)
        rectify_residual(rect)
    else:
        print("corners not found")
    # plt.scatter(corners[:, 0], corners[:, 1])
    # plt.scatter(objp[:, 0], objp[:, 1])
    # plt.scatter(rect[:, 0], rect[:, 1])
    # plt.show()

def homogeneity_check(img1, img2):
    objp = np.mgrid[0:11, 0:8].T.reshape((88, 2))
    corners = findchessboard(img1)
    h = homography(corners, objp)

    corners2 = findchessboard(img2)
    if corners2 is not None:
        rect = rectify(corners2, h)
        rectify_residual(rect)
    else:
        print("corners not found")

if __name__ == "__main__":
    homogeneity_check("data/2023-02-10/un_zhang_1.bmp", "data/2023-02-10/un_zhang_1.bmp")