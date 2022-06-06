import numpy as np
from dataclasses import dataclass
from sympy import *

@dataclass
class CV():
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

        # Q = np.zeros((7,7))
        # Q[:2, :2] = T**2*np.eye(2)
        # Q[:2, 2:4] = T*np.eye(2)
        # Q[2:4, 2:4] = T*np.eye(2)

        # Brekke 4.64
        T2 = T**2
        T3 = T**3
        Q = np.array([[T3/3, 0, T2/2, 0, 0, 0, 0],
                      [0, T3/3, 0, T2/2, 0, 0, 0],
                      [T2/2, 0, T, 0, 0, 0, 0],
                      [0, T2/2, 0, T, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])

        return Q*self.sigma**2



