from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from proposed.model.camera_model import CameraModel

import numpy as np

class Points:
    """
        _O: original image, full size before standardization
        -----------------------------
        in standardized variables:
        i: standardized image pixel
        d: radial symmetric distorted
        u: undistorted
        c: centered by cod
        1
    """
    def __init__(self, ps:np.ndarray, _cm:CameraModel):
        self.origin = np.array(ps).astype('float64')
        self.cm = _cm
        self.pi = self.origin/self.cm.scale
        self.xi = self.pi[:, 0]
        self.yi = self.pi[:, 1]

    def update_model(self):
        self.cx = self.cm.cod[0] / self.cm.scale
        self.cy = self.cm.cod[1] / self.cm.scale
        self.xicx = self.xi - self.cx
        self.yicy = self.yi - self.cy
        self.pic = np.column_stack((self.xicx, self.yicy, np.ones_like(self.xicx)))
        
        self.pdc = np.dot(self.pic, self.cm.homography.T)
        self.xdcx = self.pdc[:, 0] / self.pdc[:, 2]
        self.ydcy = self.pdc[:, 1] / self.pdc[:, 2]

        self.rd2 = self.xdcx**2 + self.ydcy**2
        h = self.cm.homography
        self.h11 = h[0, 0]
        self.h12 = h[0, 1]
        self.h21 = h[1, 0]
        self.h22 = h[1, 1]
        self.h31 = h[2, 0]
        self.h32 = h[2, 1]
    
    def undistort(self):
        self.xucx = np.array(self.xdcx)
        self.yucy = np.array(self.ydcy)
        for i in range(self.cm.degree):
            self.xucx += self.cm.dist_coeff[i, 0] * self.xdcx * self.rd2**(i + 1)
            self.yucy += self.cm.dist_coeff[i, 0] * self.ydcy * self.rd2**(i + 1)
    
    def restore_image(self):
        self.undistort()
        self.xu_O = (self.xucx + self.cx) * self.cm.scale
        self.yu_O = (self.yucy + self.cy) * self.cm.scale
        
        return np.column_stack([self.xu_O, self.yu_O])

class PlumbLine(Points):
    NEIBOUR_COUNT = 100
    def __init__(self, l:np.ndarray, _cm:CameraModel):
        super().__init__(l, _cm)
        
    def _line_norm(self, seg):
        norm_rad = 0
        if abs(seg[-1, 1] - seg[0, 1]) <= abs(seg[-1, 0] - seg[0, 0]):
            params = np.polyfit(seg[:, 0], seg[:, 1], 1)
            norm_rad = np.arctan(params[0]) + np.pi / 2
            if norm_rad > np.pi / 2:
                norm_rad -= np.pi
        else:
            params = np.polyfit(seg[:, 1], seg[:, 0], 1)
            norm_rad = -np.arctan(params[0])
        return norm_rad
    
    def neibour_line_fit(self):
        """
            returns the normal vector angle of a line
        """
        diff = np.array(self.origin)
        diff[:, 0] -= self.cm.cod[0]
        diff[:, 1] -= self.cm.cod[1]
        dist = diff[:, 0]**2 + diff[:, 1]**2
        minidx = np.argmin(dist)
        seg = self.origin[minidx - PlumbLine.NEIBOUR_COUNT:minidx + PlumbLine.NEIBOUR_COUNT+1]
        self.norm_rad = self._line_norm(seg)
        self.line_rad = self._line_norm(self.origin)
        # print("norm_rad: ", self.norm_rad)
        self.a = np.cos(self.norm_rad)
        self.b = np.sin(self.norm_rad)
        
    def line_equation(self):
        c_list = list()
        for i in range(self.cm.degree):
            c = (self.a * self.xdcx + self.b * self.ydcy) * self.rd2**(i + 1)
            c_list.append(c)
        x = np.column_stack(c_list)
        # y = -self.b * self.yd - self.a * self.xd
        y = -self.a * self.xdcx - self.b * self.ydcy
        return x, y

    def restore(self, c): 
        self.undistort()
        self.c = c
        self.res = self.a * self.xucx + self.b * self.yucy + self.c
        self.avg_res = np.mean(self.res**2)
        self.sum_res = np.sum(self.res**2)
    
    def gradient(self):
        """
            the gradient to be ADDED to the current guess of cod
        """
        factor1, factor2 = 1, 0
        for i in range(self.cm.degree):
            factor1 += self.cm.dist_coeff[i, 0] * self.rd2**(i + 1)
            factor2 += (i + 1) * self.cm.dist_coeff[i, 0] * self.rd2**i
        # factor1 = self.k1 * self.rd2 + self.k2 * self.rd2**2 + self.k3 * self.rd2**3
        # factor2 = self.k1 + 2 * self.k2 * self.rd2 + 3 * self.k3 * self.rd2**2
        # partial derivative on cx cy
        # dxu_dcx = factor1 + 2 * self.xdcx**2 * factor2
        # dyu_dcy = factor1 + 2 * self.ydcy**2 * factor2 #self.k1 * (self.ydcy * self.drd2_dcy - self.rd2)

        dxucx_dcx = factor1 + 2 * self.xdcx**2 * factor2
        dyucy_dcy = factor1 + 2 * self.ydcy**2 * factor2
        
        dyu_dcx = 2 * self.xdcx * self.ydcy * factor2 #(self.k1 + 2 * self.k2 * self.rd2 + 3 * self.k3 * self.rd2**2)
        dxu_dcy = dyu_dcx #self.xdcx * self.drd2_dcy * (self.k1 + 2 * self.k2 * self.rd2 + 4 * self.k3 * self.rd2**2)
        dxucx_dcy = dxu_dcy
        dyucy_dcx = dyu_dcx
        # ds_dcx = 2 * self.res * (self.a * dxu_dcx + self.b * dyu_dcx)
        # ds_dcy = 2 * self.res * (self.a * dxu_dcy + self.b * dyu_dcy)
        ds_dcx = 2 * self.res * (self.a * dxucx_dcx + self.b * dyucy_dcx)
        ds_dcy = 2 * self.res * (self.a * dxucx_dcy + self.b * dyucy_dcy)

        # partial derivative on homography
        factor3 = self.h31 * self.xicx + self.h32 * self.yicy + 1

        dxdcx_dh11 = self.xicx / factor3
        dxdcx_dh12 = self.yicy / factor3
        dydcy_dh21 = self.xicx / factor3
        dydcy_dh22 = self.yicy / factor3

        dxdcx_dh21 = dxdcx_dh22 = dydcy_dh11 = dydcy_dh12 = 0

        dxdcx_dh31 = -self.xicx * (self.h11 * self.xicx + self.h12 * self.yicy) / factor3**2
        dxdcx_dh32 = -self.yicy * (self.h11 * self.xicx + self.h12 * self.yicy) / factor3**2

        dydcy_dh31 = -self.xicx * (self.h21 * self.xicx + self.h22 * self.yicy) / factor3**2
        dydcy_dh32 = -self.yicy * (self.h21 * self.xicx + self.h22 * self.yicy) / factor3**2

        dx_dh = [dxdcx_dh11, dxdcx_dh12, dxdcx_dh21, dxdcx_dh22, dxdcx_dh31, dxdcx_dh32]
        dy_dh = [dydcy_dh11, dydcy_dh12, dydcy_dh21, dydcy_dh22, dydcy_dh31, dydcy_dh32]
        
        drd2_dh = [2 * self.xdcx * dx_dh[i] + 2 * self.ydcy * dy_dh[i] for i in range(6)]
        dxucx_dh = [dx_dh[i] * (factor1) + self.xdcx * drd2_dh[i] * factor2 for i in range(6)]
        dyucy_dh = [dy_dh[i] * (factor1) + self.ydcy * drd2_dh[i] * factor2 for i in range(6)]
        de_dh = [np.mean(-2 * self.res * (self.a * dxucx_dh[i] + self.b * dyucy_dh[i])) for i in range(6)]
        
        return np.mean(ds_dcx)*self.cm.scale, np.mean(ds_dcy)*self.cm.scale, de_dh
    
    def straightness(self):
        """
        calculate the mean residual of a line fit to reflect straightness of undistorted lines
        must be called after undistort

        """
        if abs(self.xu_O[0] - self.xu_O[-1]) >= abs(self.yu_O[0] - self.yu_O[-1]):
            x, y = self.xu_O, self.yu_O
        else:
            x, y = self.yu_O, self.xu_O
        params = np.polyfit(x, y, 1)

        pred = params[0] * x + params[1]
        return np.mean((pred-y)**2)
        # res = self.a * self.xu + self.b * self.yu + self.c
        # return np.mean(self.res**2)





