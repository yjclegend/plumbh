import numpy as np

def straightness(line):
    """
    calculate the mean residual of a line fit to reflect straightness of undistorted lines
    must be called after undistort

    """
    lx = line[:, 0]
    ly = line[:, 1]
    if abs(lx[0] - lx[-1]) >= abs(ly[0] - ly[-1]):
        x, y = lx, ly
    else:
        x, y = ly, lx
    params = np.polyfit(x, y, 1)

    pred = params[0] * x + params[1]
    # return np.mean((pred-y)**2)
    return np.sqrt(np.mean((pred-y)**2))