import re
import numpy as np
from dataclasses import dataclass

@dataclass
class RangeOnly():
    
    sigma: float
    m: int = 2
    state_dim: int = 6
    pdim: int =2

    def h(self, x):
        return np.array([np.linalg.norm(x[:self.pdim], 2)]).reshape((-1, 1))

    def H(self, x):
        range = self.h(x)
        if range == 0:
            return 0

        H = np.zeros((1, self.state_dim))

        H[0, :self.pdim] = (x[:self.pdim]/range).flatten()
        return H

    def R(self, x):
        R = np.array([self.sigma**2]).reshape((1, 1))
        return R