import cv2
import numpy as np

class Calibration:
    def __init__(self, use_k2=False, use_k3=False, use_decenter=False):
        self.use_k2 = use_k2
        self.use_k3 = use_k3
        self.use_decenter = use_decenter
        self.CHESSBOARD = (11, 8)
        self.objp = np.mgrid[0:11, 0:8].T.reshape((88, 2))
        # self.objp = self.objp * 100 + 1000

    def find_chessboard(self, fname):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
        ret, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(corners.shape)
        if ret == True:
            print("corners found")
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            return np.reshape(corners2, (88, 2))
    
    def calibarte(self):
        pass

    def saveparams(self, ):
        pass
    def undistort_points(self, points):
        pass
    def undistort_image(self, image):
        pass

    def homography_svd(self, pers, rect):
        assert(pers.shape[0] == rect.shape[0])
        equations = list()
        for i in range(pers.shape[0]):
            equation1 = [-pers[i, 0], -pers[i, 1], -1, 0, 0, 0, rect[i, 0]*pers[i, 0], rect[i, 0]*pers[i, 1], rect[i, 0]]
            equation2 = [0, 0, 0, -pers[i, 0], -pers[i, 1], -1, rect[i, 1]*pers[i, 0], rect[i, 1]*pers[i, 1], rect[i, 1]]
            equations.append(equation1)
            equations.append(equation2)

        equations = np.array(equations)
        print(equations.shape)
        U, singular, V_transpose = np.linalg.svd(equations)

        self.h = np.reshape(V_transpose[-1], (3, 3))
        return self.h
    
    def homography(self, pers, rect):
        assert(pers.shape[0] == rect.shape[0])
        equations = list()
        for i in range(pers.shape[0]):
            equation1 = [-pers[i, 0], -pers[i, 1], -1, 0, 0, 0, rect[i, 0]*pers[i, 0], rect[i, 0]*pers[i, 1], rect[i, 0]]
            equation2 = [0, 0, 0, -pers[i, 0], -pers[i, 1], -1, rect[i, 1]*pers[i, 0], rect[i, 1]*pers[i, 1], rect[i, 1]]
            equations.append(equation1)
            equations.append(equation2)
        equations = np.array(equations)
        A = equations[:, :-1]
        b = -1 * equations[:, -1]

        h = np.linalg.lstsq(A, b)[0]
        h = np.append(h, 1)
        self.h = np.reshape(h, (3, 3))
        return self.h
        

    def rectify(self, points):
        points_homo = np.column_stack([points, np.ones((points.shape[0], 1))])
        rectify = np.dot(self.h, points_homo.T).T
        rectify[:, 0] /= rectify[:, 2]
        rectify[:, 1] /= rectify[:, 2]
        return rectify[:, :2]