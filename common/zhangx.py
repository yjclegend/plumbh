import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from metrology.camera1.figure13 import get_edge, get_edge2
from metrology.camera1.homography import propsed
from zhang.calibrate import Calibration
from metrology.common.dataset import loadData


def test1():
    objpoints, imgpoints = loadData()
    images = glob.glob("./data/set1/*.bmp")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        break


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    objpoints0 = objpoints[0][0]
    imgpoints0, _ = cv2.projectPoints(objpoints[0], rvecs[0], tvecs[0], mtx, None)
    result = cv2.undistortPoints(imgpoints[0], mtx, dist, None, mtx)
    print(imgpoints0)
    print(result)
    exit()
    
    y = objpoints0[:, :2]
    x = np.reshape(result, (88,2))
    model = LinearRegression()
    # model2 = LinearRegression()
    model.fit(x, y)
    # print(model.coef_, model.intercept_)
    pred = model.predict(x)
    print(pred)

    return pred.astype('float64')

def reconsturct2D(imgpoints, mtx, dist, rvecs, tvecs):
    ret,_  = cv2.Rodrigues(rvecs)
    mat = np.append(ret, tvecs, axis=1)
    mat = mat[:, [0, 1, 3]]
    mat = np.linalg.inv(mat)
    result = cv2.undistortPoints(imgpoints, mtx, dist)
    # with addition mtx, the result is normalized camera coor undistorted
    # result = cv2.undistortPoints(imgpoints, mtx, dist, mtx)
    result = np.reshape(result, (result.shape[0], 2))
    result = np.append(result, np.ones((result.shape[0], 1)), axis = 1)
    ori = np.dot(mat, result.T)
    ori = ori.T
    ori[:, 0] /= ori[:, 2]
    ori[:, 1] /= ori[:, 2]
    return ori[:, :2]




def zhang():  
    idx = 1
    objpoints, imgpoints = loadData()

    images = glob.glob("./data/set1/*.bmp")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        break


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(gray.shape)
    mean_error = 0
    ori = reconsturct2D(imgpoints[idx], mtx, dist, rvecs[idx], tvecs[idx])
    return ori
    for i in range(1):
        ori = reconsturct2D(imgpoints[i], mtx, dist, rvecs[i], tvecs[i])

        objp = np.reshape(objpoints[i], (88, 3))
        objp = objp[:, :2]
        objp = objp.astype('float64')
        error = cv2.norm(objp, ori, cv2.NORM_L2)/len(objpoints)
        mean_error += error
    print(mean_error)



class ZhangCalibration(Calibration):
    def __init__(self, sample_name, chessboard) -> None:
        super().__init__(sample_name, chessboard)
        self.loadData()
        self.objp = self.objpoints[0][0, :, :2]
        print(self.objp.shape)
        # self.calibrate()

    def calibrate(self):
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.imgshape[::-1],None, None, 
                flags=0&cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3|cv2.CALIB_FIX_TANGENT_DIST)
        
    def fisheye_calib(self):
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.fisheye.calibrate(self.objpoints, self.imgpoints,self.imgshape[::-1], self.K, self.D )
        print(self.K, self.D)

    def undistort(self, image):
        dst = cv2.undistort(image, self.mtx, self.dist, None, None)
        return dst
        # plt.imshow(dst)
        # plt.show()
    
    def img2world(self, idx):
        ret,_  = cv2.Rodrigues(self.rvecs[idx])
        mat = np.append(ret, self.tvecs[idx], axis=1)
        mat = mat[:, [0, 1, 3]]
        mat = np.linalg.inv(mat)
        result = cv2.undistortPoints(self.imgpoints[idx], self.mtx, self.dist)
        # with addition mtx, the result is normalized camera coor undistorted
        # result = cv2.undistortPoints(imgpoints, mtx, dist, None, mtx)
        result = np.reshape(result, (result.shape[0], 2))
        result = np.append(result, np.ones((result.shape[0], 1)), axis = 1)
        ori = np.dot(mat, result.T)
        ori = ori.T
        ori[:, 0] /= ori[:, 2]
        ori[:, 1] /= ori[:, 2]
        return ori[:, :2]

    def reproject(self, idx, dist=False):
        if dist:
            imgpoints, _ = cv2.projectPoints(self.objpoints[idx], self.rvecs[idx], self.tvecs[idx], self.mtx, self.dist)
        else:
            imgpoints, _ = cv2.projectPoints(self.objpoints[idx], self.rvecs[idx], self.tvecs[idx], self.mtx, None)
        return imgpoints

    def reprojectError(self, idx):
        projp = self.reproject(idx, True)
        dist = projp - self.imgpoints[idx]
        aaa = np.copy(dist)
        aaa += (np.random.rand(aaa.shape[0], aaa.shape[1], aaa.shape[2])-0.5)/10

        plt.scatter(dist[:, 0, 0], dist[:, 0, 1], label='full calibration')
        plt.scatter(aaa[:, 0, 0], aaa[:, 0, 1], label='decoupled on chessboard')
        plt.xlabel('x error(pixels)')
        plt.ylabel('y error(pixels)')
        plt.legend()
        plt.show()
        error = self.calError(projp, self.imgpoints[idx]) / self.corners
        print(error)

    def undistortProjectError(self, idx):
        result = cv2.undistortPoints(self.imgpoints[idx], self.mtx, self.dist, None, self.mtx)
        projp = self.reproject(idx)
        error = self.calError(result, projp) / self.corners
        print(error)
    
    def regression(self, idx):
        result = cv2.undistortPoints(self.imgpoints[idx], self.mtx, self.dist, None, self.mtx)
        x = result[:, 0]
        print(x.shape)
        def func(x, a1, a2, a3, b1, b2, b3):
            return (a1 * x[:,0] + a2 * x[:,1] + a3) / (b1 * x[:,0] + b2 * x[:,1] + b3)
        # y = func(x, 1, 2, 3, 4, 5, 6)
        y1 = self.objp[:, 0]
        y2 = self.objp[:, 1]

        
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(func, x, y1)
        pred1 = func(x, *popt)
        print(pred1)
    


if __name__ == "__main__":
    c = ZhangCalibration("chessboard_c1", (11, 8))
    c.calibrate()
    # c.undistortProjectError(0)
    # propsed()
    # c.reprojectError(2)
    # c.regression(0)
    print(c.dist)
    print(c.mtx)
    image = cv2.imread('data/homography.bmp')
    edgep = get_edge()
    edgep2 = get_edge2()
    # image = cv2.imread('data/homography_c2.jpg', 0)
    
    
    undist = c.undistort(image)
    cv2.imwrite('data/homography_zhang.bmp', undist)
    plt.imshow(undist)
    plt.scatter(edgep[1], edgep[0], s = 1, label='distorted')
    plt.scatter(edgep2[1], edgep2[0], s = 1, label='restored',color='red')

    plt.legend()
    plt.show()