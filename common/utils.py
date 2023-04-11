import os
import numpy as np

def enum_path(root):
    files = os.listdir(root)
    file_paths = list()
    for i in range(len(files)):
        file_path = os.path.join(root, files[i])
        file_paths.append(file_path)
    return file_paths


def fit_line(sample):
    equations = np.append(sample, np.ones((sample.shape[0], 1)), axis=1)
    U, singular, V_transpose = np.linalg.svd(equations)
    vector = V_transpose[-1]
    norm_vec = vector[:2]
    c = vector[2]
    norm = np.linalg.norm(norm_vec)
    norm_vec /= norm
    c /= norm
    return norm_vec, c
    res = norm_vec[0] * sample[:, 0] + norm_vec[1] * sample[:, 1] + c
    return np.sum(res**2)

def cal_residual(sample, norm_vec, c):
    res = norm_vec[0] * sample[:, 0] + norm_vec[1] * sample[:, 1] + c
    return np.sum(res**2)