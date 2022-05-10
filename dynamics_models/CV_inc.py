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
    n: int=2

    def f(self, x, u, T):

        F = np.eye(self.n*3)
        F[:self.n, self.n:self.n*2] = np.eye(self.n)*T

        return F@x

    def F(self, x, u, T):

        F = np.eye(self.n*3)
        F[:self.n, self.n:self.n*2] = np.eye(self.n)*T

        return F
    
    def Q(self, x, u, T):

        if self.n != 2:
            raise NotImplemented

        Q = np.zeros((self.n*3, self.n*3))
        Q[:self.n, :self.n] = T**2*np.eye(self.n)
        Q[:self.n, self.n:self.n*2] = T*np.eye(self.n)
        Q[self.n:self.n*2, :self.n] = T*np.eye(self.n)
        Q[self.n:self.n*2, self.n:self.n*2] = np.eye(self.n)

        return Q*self.sigma**2

def compute_Q():
    T = symbols("T")
    sa = symbols("sa")
    Qv = zeros(2*2, 2*2)
    Qv[2:, 2: ] = eye(2) #only noise on velocity
    
    print("Qv", Qv)
    F = eye(4)
    F[0:2, 2:4] = T*eye(2)


   

    print(F)

    Q = F@Qv@F.T#*sa**2
    for row in range(n*2):
        for col in range(n*2):
            print(Q[row, col], end=" ")
        print()


if __name__=='__main__':
    n = 2
    cv = CV(
        n=n, 
        sigma=1,
    )
    x0 = np.ones((n*2))
    T = 0.1
    compute_Q()
    print("x_nxt", cv.f(x0, T))
    print("F", cv.F(x0, T))
    print("Q", cv.Q(x0, T))



