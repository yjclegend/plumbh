import numpy as np
import cv2, os
import glob
import pickle

import matplotlib.pyplot as plt

from common.chessboard import findchessboard, prepare_chessboard
class Calibration:
    def __init__(self, sample_name, chessboard) -> None:
        self.name = sample_name
        # self.path = "./data/" + sample_name
        self.chessboard = chessboard
        self.corners = chessboard[0] * chessboard[1]

    def prepareData(self):
        criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
        objp = np.zeros((1, self.chessboard[0] * self.chessboard[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.chessboard[0], 0:self.chessboard[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        images = glob.glob("./data/%s/*.jpg" % (self.name))
        image_shape = None
        for fname in images:
            print(fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('img',thresh)
            # cv2.waitKey(0)
            if not image_shape:
                image_shape = gray.shape
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard, cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
            print(corners.shape)
            if ret == True:
                print("corners found")
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, self.chessboard, corners2, ret)
            cv2.imshow('img',img)
            cv2.waitKey(0)
        dataset = dict()
        dataset["obj"] = objpoints
        dataset["img"] = imgpoints
        dataset['shape'] = image_shape
        f = open(self.name + '.pkl', "wb")
        pickle.dump(dataset, f)
        f.close()

    def loadData(self):
        f = open("./data/saved/%s.pkl" % (self.name), "rb")
        dataset = pickle.load(f)
        self.objpoints = dataset["obj"]
        self.imgpoints = dataset["img"]
        self.imgshape = dataset['shape']

    
    def calError(self, a, b):
        return cv2.norm(a.astype('float64'), b.astype('float64'), cv2.NORM_L2)

def calibration(path, size):
    objp = np.zeros((1, size[0] * size[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        corners = findchessboard(fname=fpath)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (3840, 2748), None, None)
    if ret:
        coeffs = dict()
        coeffs['mtx'] = mtx
        coeffs['dist'] = dist
        coeffs['rvecs'] = rvecs
        coeffs['tvecs'] = tvecs
        f = open(path + '/coeffs.pkl', "wb")
        pickle.dump(coeffs, f)
        return coeffs

def undistort(im_path, coeff_path):
    img = cv2.imread(im_path, 0)
    f = open(coeff_path, "rb")
    coeffs = pickle.load(f)
    dst = cv2.undistort(img, coeffs['mtx'], coeffs['dist'], None, None)
    names = os.path.split(im_path)
    print(names)
    cv2.imwrite(os.path.join(names[0], "un_zhang_" + names[1]), dst)
    plt.imshow(dst)
    plt.show()


