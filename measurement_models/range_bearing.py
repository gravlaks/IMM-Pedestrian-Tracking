import re
import numpy as np
from dataclasses import dataclass

@dataclass
class RangeBearing():
    
    sigma: float
    m: int = 2
    state_dim: int = 2

    def h(self, x):
        p = x[:2]
        # if p[0] == 0:
        #     meas = np.vstack((np.linalg.norm(p), np.sign(p[1])*np.pi/2 ))

        # else:
        meas = np.vstack((np.linalg.norm(p), 
            np.arctan2(p[1], p[0])))

        if meas[0]>100000000 or meas[1]>10000000000:
            import pdb;pdb.set_trace()

        return meas

    def H(self, x):
        #http://www.cs.cmu.edu/~16385/s17/Slides/16.4_Extended_Kalman_Filter.pdf
        p = x[:2]
        # if p[0] == 0:
        #     theta = np.sign(p[1])*np.pi/2
        # else:
        #     theta = np.arctan(p[1]/p[0])
        theta = np.arctan2(p[1], p[0])
        r = np.linalg.norm(p)
        H = np.zeros((2, self.state_dim))
        H[0, 0] = np.cos(theta)
        H[0, 1] = np.sin(theta)
        if p[0] == 0:
            H[0, 0] = 0
            H[0, 1] = 1
        if r == 0: 
            return H
        
        H[1, 0] = -np.sin(theta)/r
        H[1, 1] = np.cos(theta)/r
      
        return H

    def R(self, x=None):
        R = np.eye(self.m)*self.sigma**2
        return R