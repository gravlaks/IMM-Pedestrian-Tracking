import re
import numpy as np
from dataclasses import dataclass

@dataclass
class RangeOnly():
    
    sigma: float
    m: int = 2
    n: int = 2

    def h(self, x):
        

        return np.array([np.linalg.norm(x[:self.n], 2)]).reshape((-1, 1))

    def H(self, x):
        p = x[:self.n]
        range = np.linalg.norm(x[:self.n])
        if range == 0:
            return np.zeros((1, self.n*2))
        H = np.zeros((1, self.n*2))
        

        H[0, :self.n] = (p/range).flatten()
        return H

    def R(self, x):
        R = np.array([self.sigma**2]).reshape((1, 1))
        return R