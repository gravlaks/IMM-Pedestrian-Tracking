import numpy as np
from dataclasses import dataclass
from sympy import *

@dataclass
class CV_7dim():
    """
    n is dimension of state space
    sigma is noise 
    """
    sigma: float

    def f(self, x, u, T):
        F = np.zeros((7, 7))
        F[:4, :4] = np.eye(4)
        F[:2, 2:4] = np.eye(2)*T

        return F@x

    def F(self, x, u, T):

        F = np.zeros((7, 7))
        F[:4, :4] = np.eye(4)
        F[:2, 2:4] = np.eye(2)*T

        return F
    
    def Q(self, x, u, T):

   

        Q = np.zeros((7,7))
        Q[:2, :2] = T**2*np.eye(2)
        Q[:2, 2:4] = T*np.eye(2)
        Q[2:4, 2:4] = T*np.eye(2)

        return Q*self.sigma**2



