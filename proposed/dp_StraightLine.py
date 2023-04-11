import os, time
import numpy as np

from calibration.Calibration import Calibration

from imagedata.realimage.PolygonImage import PolygonImage
from imagedata.realimage.samples import LineSegment
from common.utils import cal_residual, fit_line


class StraighLine(Calibration):
    def __init__(self, use_k2=False, use_k3=False, use_decenter=False):
        super().__init__(use_k2, use_k3, use_decenter)
        self.segments:list[LineSegment] = list()
    
    def find_segments(self, root):
        files = os.listdir(root)
        for i in range(len(files)):
            file_path = os.path.join(root, files[i])
            image = PolygonImage(file_path)
            self.segments.extend(image.segments)
        self.segments = self.segments[:12]

class StraightLineCOD(StraighLine):
    def __init__(self, use_k2=False, use_k3=False, use_decenter=False):
        super().__init__(use_k2, use_k3, use_decenter)
        self.segments:list[LineSegment] = list()
        self.best_params = None
        
    
        # self.segments = self.segments[8:12]

    def __estimate_with_cod(self, cod):
        cod = np.array(cod).reshape((1, 2))
        x_list = list()
        y_list = list()
        
        for i in range(len(self.segments)):

            seg = self.segments[i]
            seg.set_cod(cod)
            seg.norm_to_cod()
            # seg.norm_blf()

            sx, sy = seg.build_equation(self.use_k2, self.use_k3, self.use_decenter) 
            sparse = np.zeros((sx.shape[0], len(self.segments)))
            sparse[:, i] = 1
            sx = np.column_stack([sx, sparse])
            sy = sy[:, np.newaxis]
            x_list.append(sx)
            y_list.append(sy)
        
        x = np.row_stack(x_list)
        y = np.row_stack(y_list)
        params = np.linalg.lstsq(x, y)[0]
        cs = params[-len(self.segments):]
        distortion = params[:-len(self.segments)]
        # print(distortion)
        res = 0

        for i in range(len(self.segments)):
            seg = self.segments[i]
            c = cs[i, 0]
            restored = seg.restore(c, distortion, self.use_k2, self.use_k3, self.use_decenter)
            res += seg.cal_residual()
            # norm, cc = fit_line(restored)
            # res += cal_residual(restored, norm, cc)

        return res, params

    def calibrate(self, cod, sr = 5):
        cx, cy = cod
        minres = -1
        best_i, best_j = 0, 0
        # z = np.zeros((sr*2, sr*2, 1))
        for i in range(-sr, sr):
            for j in range(-sr, sr):
                cod = (cx + i, cy + j)
                res, distortion = self.__estimate_with_cod((cx + i, cy + j))
                # z[sr + i, sr + j] = res
                if minres < 0 or res < minres:
                    minres = res
                    best_i, best_j = i, j
                    self.best_params = distortion
        # np.save("zzz1.npy", z)
        print("min res:", minres)
        point_count = 0
        cs = self.best_params[-len(self.segments):]
        distortion = self.best_params[:-len(self.segments)]
        for i in range(len(self.segments)):
            seg = self.segments[i]
            c = cs[i, 0]
            seg.restore(c, distortion, self.use_k2, self.use_k3, self.use_decenter)
            point_count += self.segments[i].sample.shape[0]
        print(point_count)
        print("min averge res:", minres/point_count)
        self.cod = cx + best_i, cy + best_j
        print("best cod", self.cod)
    
