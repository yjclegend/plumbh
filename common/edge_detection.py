import cv2
import numpy as np

import matplotlib.pyplot as plt
def subpixel(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # gx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=cv2.FILTER_SCHARR)
    # gy = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=cv2.FILTER_SCHARR)
    gy = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3).astype('float64')
    gx = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3).astype('float64')
    g = np.sqrt(gx**2 + gy**2)
    edges = np.zeros_like(image)
    THRESH = 100
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            intensity = g[i, j]
            if intensity > THRESH:
                if i == 0 or i == g.shape[0] - 1 or j == 0 or j == g.shape[1] - 1:
                    edges[i, j] == 255
                else:
                    a, b, c = 0, intensity, 0
                    dx, dy = 0, 0
                    if gx[i, j] == 0:
                        dx, dy = 0, np.sign(gy[i, j])
                        a = g[i, j - dy]
                        c = g[i, j + dy]
                    elif gy[i, j] == 0:
                        dx, dy = np.sign(gx[i, j]), 0
                        a = g[i - dx, j]
                        c = g[i + dx, j]
                    elif abs(gx[i, j]) == abs(gy[i, j]):
                        dx, dy = np.sign(gx[i, j]), np.sign(gy[i, j])
                        a = g[i - dx, j - dy]
                        c = g[i + dx, i + dy]
                    elif abs(gx[i, j]) > abs(gy[i, j]):
                        dx = np.sign(gx[i, j])
                    else:
                        pass
                   
    return edges

def costom_canny(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # gx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=cv2.FILTER_SCHARR)
    # gy = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=cv2.FILTER_SCHARR)
    gy = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3).astype('float64')
    gx = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3).astype('float64')
    g = np.sqrt(gx**2 + gy**2)
    plt.imshow(blur)
    # edges = np.zeros_like(image)
    edge_points = list()
    THRESH = 80
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            intensity = g[i, j]
            if intensity > THRESH:
                if i == 0 or i == g.shape[0] - 1 or j == 0 or j == g.shape[1] - 1:
                    # edges[i, j] == 255
                    edge_points.append((i, j))
                else:
                    ax, ay = 0, 0
                    if abs(gx[i, j]) >= abs(gy[i, j]):
                        ax = 1
                    else:
                        ay = 1
                    if gx[i, j] != 0:
                        tan = gy[i, j] / gx[i, j]
                        if abs(tan) > np.tan(np.radians(22.5)) and abs(tan) < np.tan(np.radians(67.5)):
                            ax, ay = 1, int(np.sign(tan))
                    if intensity >= g[i - ax, j - ay] and intensity >= g[i + ax, j + ay]:
                        # edges[i, j] = 255
                        edge_points.append((i, j))
    return edge_points