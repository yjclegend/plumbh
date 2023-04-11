import cv2
import numpy as np
import matplotlib.pyplot as plt

from common.zebra import LineSegment

def test1():
    image = cv2.imread('data/fisheye.jpg', 0)
    _, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(image, 100, 150)
    plt.imshow(thresh)
    plt.show()

if __name__ == "__main__":
    # test_prepare_line()
    test1()