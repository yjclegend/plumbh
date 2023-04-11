import pickle
import numpy as np

from proposed.model.samples import PlumbLine, Points



class CameraModel:
    def __init__(self, _scale, _cod, deg=3, homo=False, clr=10, hlr=0.001):
        self.scale = _scale
        self.cod = _cod
        self.degree = deg
        self.dist_coeff: np.ndarray
        self.homo = homo
        self.homography = np.eye(3)
        self.homo_list = [self.homography.flatten()]
        self.clr = clr
        self.hlr = hlr
        self.residual = list()

    def estimate(self, lines:list[PlumbLine], iters=60, cheating=None):
        self.cod_list = [self.cod]
        self.cheating = cheating
        self.residual = list()
        for i in range(iters):
            # print("iteration: ", i)
            self.estimate_distortion(lines)
            self.cod_list.append(self.cod)
    
    def estimate_distortion(self, lines:list[PlumbLine]):
        line_num = len(lines)
        x_list = list()
        y_list = list()
        for i in range(line_num):
            line = lines[i]
            line.update_model()
            if self.cheating is not None:
                line.norm_rad = self.cheating[i]
                line.a = np.cos(line.norm_rad)
                line.b = np.sin(line.norm_rad)
            else:
                line.neibour_line_fit()
            x, y = line.line_equation()
            sparse = np.zeros((x.shape[0], line_num))
            sparse[:, i] = 1
            sx = np.column_stack([x, sparse])
            sy = y[:, np.newaxis]
            x_list.append(sx)
            y_list.append(sy)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        params = np.linalg.lstsq(x, y)[0]
        cs = params[-line_num:]
        self.dist_coeff = params[:-line_num]
        self._update_model(lines, cs)
        # print(cs)
        # print(distortion)
    
    def _update_model(self, lines:list[PlumbLine], cs):
        sdx, sdy = 0, 0
        cx, cy = self.cod
        sde_dh = [0] * 6
        res = 0
        for i in range(len(lines)):
            line = lines[i]
            line.restore(cs[i, 0])
            res += np.sqrt(line.avg_res)
            dx, dy, de_dh = line.gradient()
            sdx += dx
            sdy += dy
            sde_dh = [sde_dh[i] + de_dh[i] for i in range(6)]

        self.residual.append(self.scale * res/len(lines))
        # update COD
        cx += self.clr * sdx
        cy += self.clr * sdy
        self.cod = (cx, cy)
        if self.homo:
        # update homography
            mat_dedh = np.array(sde_dh).reshape((3, 2))
            mat_dedh = np.column_stack((mat_dedh, np.array([0, 0, 0])))
        # self.mat_dedh = mat_dedh
        # # mat_dedh[0, 0] = 0
            self.homography += mat_dedh * self.hlr
            self.homo_list.append(self.homography.flatten())
        # print(self.homography)
        # exit()
    
    def undistort_points(self, points):
        points = Points(points, self)
        points.update_model()
        undist = points.restore_image()
        return undist
        # sc = np.array(points)
        # sc[:, 0] -= self.cod[0]
        # sc[:, 1] -= self.cod[1]
        # sc /= self.scale
        # rd2 = sc[:, 0]**2 + sc[:, 1]**2
        # for i in range(self.degree):
        #     sc[:, 0] += self.dist_coeff[i, 0] * sc[:, 0] * rd2**(i + 1)
        #     sc[:, 1] += self.dist_coeff[i, 0] * sc[:, 1] * rd2**(i + 1)
        # sc *= self.scale
        # sc[:, 0] += self.cod[0]
        # sc[:, 1] += self.cod[1]
        # return sc

        
    def save_param(self, name):
        params = dict()
        params['cod'] = self.cod
        params['dist'] = self.dist_coeff
        params['scale'] = self.scale
        params['homo'] = self.homography 
        f = open('data/'+ name + '.pkl', "wb")
        pickle.dump(params, f)

    def load_param(self, name):
        f = open("./data/%s.pkl" % (name), "rb")
        params = pickle.load(f)
        self.cod = params['cod']
        self.dist_coeff = params['dist']
        self.scale = params['scale']
        self.homography = params['homo']


