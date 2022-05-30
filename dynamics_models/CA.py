import numpy as np
from dataclasses import dataclass
from sympy import *

@dataclass
class CA():
    """
    n is dimension of state space
    sigma is noise 
    """
    sigma: float
    n: int=2

    def f(self, x, u, T):

        F = self.F(x, u, T) 

        # compensating
        w = x[-1]
        dx = x[2]
        dy = x[3]
        ddx = x[4]
        ddy = x[5]
        denom = dx**2 + dy**2
        #if not denom == 0:
        #F[6,6] = (-ddx*dy + ddy*dx)/denom
        out = F@x
        if not denom == 0:
            out[6] = (-ddx*dy + ddy*dx)/denom
        return out 

    def F(self, x,u,T):
        F = np.zeros((7, 7))
        F[:6, :6] = np.eye(self.n*3)

        x = x.reshape(-1, 1)
        dx = x[2][0]
        dy = x[3][0]
        ddx = x[4][0]
        ddy = x[5][0]

        if u is None:
            F[:self.n, self.n:self.n*2] = np.eye(self.n)*T
            F[:self.n, self.n*2:self.n*3] = np.eye(self.n)*T**2/2
            F[self.n:self.n*2, self.n*2:self.n*3] = np.eye(self.n)*T
        else:
            raise Exception('u is not None')

        denom = dx**2 + dy**2

        if not denom == 0:

            F[6] = np.array([[0, 
                              0, 
                              ddy/denom - (ddy*dx-ddx*dy)*2*dx/denom**2, 
                              -ddx/denom + (ddx*dy-ddy*dx)*2*dy/denom**2, 
                              -dy/denom,
                              dx/denom,
                              0]])

        return F
    
    def Q(self, x,u,  T):
        if self.n!=2:
            raise NotImplemented

        #https://www.kalmanfilter.net/covextrap.html#withQ
        # Q = np.eye(self.n*3)
        # Q[:self.n, :self.n] = T**4/4.*np.eye(self.n)
        # Q[:self.n, self.n:self.n*2] = T**3/2.*np.eye(self.n)
        # Q[:self.n, self.n*2:] = T**2/2.*np.eye(self.n)

        # Q[self.n:self.n*2, :self.n] = T**3/2.*np.eye(self.n)
        # Q[self.n:self.n*2, self.n:self.n*2] = T**2*np.eye(self.n)
        # Q[self.n:self.n*2, self.n*2:] = T*np.eye(self.n)       

        # Q[self.n*2:, :self.n] = T**2/2.*np.eye(self.n)
        # Q[self.n*2:, self.n:self.n*2] = T*np.eye(self.n)
        # Q[self.n*2:, self.n*2:] = np.eye(self.n)  

        T5 = T**5
        T4 = T**4
        T3 = T**3
        T2 = T**2

        Q = np.zeros((7, 7))
        Q[:6, :6] = np.array(([[T5/20, 0.0  , T4/8, 0.0 , T3/4, 0.0 ],
                                [0.0  , T5/20, 0.0 , T4/8, 0.0 , T3/4],
                                [T4/8 , 0.0  , T3/3, 0.0 , T2/2, 0.0 ],
                                [0.0  , T4/8 , 0.0 , T3/3, 0.0 , T2/2],
                                [T3/6 , 0.0  , T2/2, 0.0 , T   , 0.0 ],
                                [0.0  , T3/6 , 0.0 , T2/2, 0.0 , T   ]]))

        #Q[4:, 4:]  = np.eye(2)*0.00001
        return Q*self.sigma**2





