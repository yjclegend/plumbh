import cv2, glob
import numpy as np

from calibration.Calibration import Calibration
from common.utils import enum_path


class ZhangCalibration(Calibration):
    def __init__(self, sample_name, chessboard):
        self.name = sample_name
        self.chessboard = chessboard
        self.image_shape = 0

    def _chessboar_samples(self):
        criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
        objp = np.zeros((1, self.chessboard[0] * self.chessboard[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.chessboard[0], 0:self.chessboard[1]].T.reshape(-1, 2)

        self.objpoints = []
        self.imgpoints = []
        images = glob.glob("./data/%s/*.bmp" % (self.name))
        for fname in images:
            
            print(fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not self.image_shape:
                self.image_shape = gray.shape

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard, cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
            print(corners.shape)
            if ret == True:
                print("corners found")
                self.objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                self.imgpoints.append(corners2)
                # Draw and display the corners
            #     img = cv2.drawChessboardCorners(img, self.chessboard, corners2, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
    def calibrate(self):
        self._chessboar_samples()
        print(self.image_shape)
        # ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_shape[::-1],None, None, 
        #         flags=cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3|cv2.CALIB_FIX_TANGENT_DIST)
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_shape[::-1],None, None, 
                flags=0&cv2.CALIB_FIX_TANGENT_DIST)
        print(self.mtx)
    
    def undistort(self, path):
        image = cv2.imread(path, 0)
        dst = cv2.undistort(image, self.mtx, self.dist, None, None)
        return dst
        
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    c = ZhangCalibration("chessboard_c1", (11, 8))
    c.calibrate()
    paths = enum_path("data/testcase/param_estimate1/")

    for i in range(len(paths)):
        undist = c.undistort(paths[i])
        cv2.imwrite("%d.bmp" % (i), undist)