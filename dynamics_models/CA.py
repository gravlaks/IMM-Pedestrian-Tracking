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

        return F@x

    def F(self, x,u,T):

        F = np.eye(self.n*3)

        if u is None:
            F[:self.n, self.n:self.n*2] = np.eye(self.n)*T
            F[:self.n, self.n*2:] = np.eye(self.n)*T**2/2
            F[self.n:self.n*2, self.n*2:] = np.eye(self.n)*T
        else:
            F[:self.n, self.n:self.n*2] = np.eye(self.n)*T
            F[:self.n, self.n*2:] = u@np.eye(self.n)*T**2/2
            F[self.n:self.n*2, self.n*2:] = u@np.eye(self.n)*T

        return F
    
    def Q(self, x,u,  T):
        if self.n!=2:
            raise NotImplemented

        #https://www.kalmanfilter.net/covextrap.html#withQ
        Q = np.eye(self.n*3)
        Q[:self.n, :self.n] = T**4/4.*np.eye(self.n)
        Q[:self.n, self.n:self.n*2] = T**3/2.*np.eye(self.n)
        Q[:self.n, self.n*2:] = T**2/2.*np.eye(self.n)

        Q[self.n:self.n*2, :self.n] = T**3/2.*np.eye(self.n)
        Q[self.n:self.n*2, self.n:self.n*2] = T**2*np.eye(self.n)
        Q[self.n:self.n*2, self.n*2:] = T*np.eye(self.n)       

        Q[self.n*2:, :self.n] = T**2/2.*np.eye(self.n)
        Q[self.n*2:, self.n:self.n*2] = T*np.eye(self.n)
        Q[self.n*2:, self.n*2:] = np.eye(self.n)  

        #Q[4:, 4:]  = np.eye(2)*0.00001
        return Q*self.sigma**2

def compute_Q():
    T = symbols("T")
    sa = symbols("sa")
    Qa = zeros(3*2, 3*2)
    Qa[4, 4] = 1
    Qa[5, 5]= 1
    
    F = eye(6)
    F[0:2, 2:4] = T*eye(2)

    F[0:2, 4:] = T**2/2.*eye(2)

    F[2:4, 4:] = T*eye(2)

   

    #print(F)

    Q = F@Qa@F.T#*sa**2
    for row in range(n*3):
        for col in range(n*3):
            print(Q[row, col], end=" ")
        print()
    #print(Q)



if __name__=='__main__':
    n = 2
    cv = CA(
        n=n, 
        sigma=1,
    )
    x0 = np.ones((n*3))
    T = 0.1
    print("x_nxt", cv.f(x0, T))
    print("F", cv.F(x0, T))

    compute_Q()
    print("Q", cv.Q(x0, T))



