import re
import numpy as np
from dataclasses import dataclass

@dataclass
class RangeBearing():
    
    sigma: float
    m: int = 2
    n: int = 2

    def h(self, x):
        p = x[:self.n]


        return np.vstack(
            (np.linalg.norm(p), 
            np.arctan(p[1]/p[0])
            )
        )

    def H(self, x):
        #http://www.cs.cmu.edu/~16385/s17/Slides/16.4_Extended_Kalman_Filter.pdf
        p = x[:self.n]

        theta = np.arctan(p[1]/p[0])
        r = np.linalg.norm(p)
        H = np.zeros((2, self.n*2))
        H[0, 0] = np.cos(theta)
        H[0, 1] = np.sin(theta)
        if r == 0: 
            return H
        H[1, 0] = -np.sin(theta)/r
        H[1, 1] = np.cos(theta)/r
      
        return H

    def R(self, x):
        R = np.diag([self.sigma**2, self.sigma**2])
        return R