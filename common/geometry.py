import numpy as np
import cv2
def distancePoint(pa, pb):
    diff = pa - pb
    return np.sqrt(diff[0]**2 + diff[1]**2)


def twoPointCosSin(p1, p2):
    d = np.linalg.norm(p1 - p2)
    costheta = (p2[0] - p1[0]) / d
    sintheta = (p2[1] - p1[1]) / d
    return costheta, sintheta

def perspective():
    import cv2
    pts1 = np.float32([[0, 260], [640, 260],
                       [0, 400], [640, 400]])
    pts2 = np.float32([[50, 20], [500, 20],
                       [50, 640], [500, 640]])
     
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    print(matrix)

def rod_mat2angles():
    rad1 = 0#np.pi / 6  #z
    rad2 = np.pi / 6  #y
    rad3 = np.pi / 6  #x
    mat1 = [np.cos(rad1), -1 * np.sin(rad1), 0, np.sin(rad1), np.cos(rad1), 0, 0, 0, 1]
    mat2 = [np.cos(rad2), 0, -1 * np.sin(rad2), 0, 1, 0, np.sin(rad2), 0, np.cos(rad2)]
    mat3 = [1, 0, 0, 0, np.cos(rad3), -1 * np.sin(rad3), 0, np.sin(rad3), np.cos(rad3)]
    mat1 = np.reshape(np.array(mat1), (3, 3))
    mat2 = np.reshape(np.array(mat2), (3, 3))
    mat3 = np.reshape(np.array(mat3), (3, 3))
    mat = np.dot(mat2, mat3)
    mat = np.dot(mat1, mat)
    rvecs = cv2.Rodrigues(mat)
    print(rvecs[0])

def rod_angles2mat():
    rad1 = 0#np.pi / 6  #z
    rad2 = np.pi / 60  #y
    rad3 = 0#np.pi / 60  #z
    rvecs = np.array([rad1, rad2, rad3])
    rvecs = np.reshape(rvecs, (3, 1))
    
    mat = cv2.Rodrigues(rvecs)
    print(mat[0])
# rod_mat2angles()
# rod_angles2mat()

if __name__ == "__main__":
    # rod_mat2angles()
    rod_angles2mat()
    # perspective()