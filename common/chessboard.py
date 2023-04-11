import cv2, os
import numpy as np

def prepare_chessboard(path, size=(11, 8)):
    files = os.listdir(path)
    corners = list()
    objp = np.zeros((1, size[0] * size[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    for f in files:
        print(f)
        image = cv2.imread(os.path.join(path, f), 0)
        corners = findchessboard(image)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints

def findchessboard(image, size=(11, 8)):
    if image is None:
        return None
    criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
    ret, corners = cv2.findChessboardCorners(image, size, cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        print("corners found")
        corners2 = cv2.cornerSubPix(image, corners, (11,11),(-1,-1), criteria)
        return corners2
    else:
        return None

def test_findchessboard():
    img = cv2.imread('data/chessboard.bmp', 0)
    corners = findchessboard(img)
    print(corners)


if __name__ == '__main__':
    # img = cv2.imread('data/2023-02-10/1.bmp')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape[::-1])
    # findchessboard('data/2023-02-10/1.bmp')
    objpoints, imgpoints = prepare_chessboard('data/chess202303211245/')
   
    # test_findchessboard()