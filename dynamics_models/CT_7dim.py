from tkinter import W
import numpy as np
from dataclasses import dataclass

@dataclass
class CT_7dim():
    """
    n is dimension of state space
    sigma is noise 
    """
    sigma_a: float
    sigma_w: float
    n: int=2

    def f(self, x, u, T):
        w = x[-1]
        T = np.zeros((7, 7))
        T[:4, :4] = np.eye(4)
        T[2:4, :4] = np.array([
            [np.sin(T*w)/w,(-1+np.cos(T*w))/w],
            [(1-np.cos(T*w))/w, np.sin(T*w)/w],
            [np.cos(T*w), -np.sin(T*w)],
            [np.sin(T*w), np.cos(T*w)]
        ])
        T[6, 6] = 1
        return T@x

    def F(self, x, u, T):
        w = x[-1]
        Jac = np.zeros((7, 7))
        Jac[:4, :4] = np.eye(4)
        xdot, ydot = x[2], x[3]
        if w == 0:
            Jac[0:2, 2:4] = np.eye(2)*T
            Jac[:4, 6] = np.array([-T**2*ydot/2,T**2*xdot/2, -T*ydot, T*xdot]).reshape((-1, 1))
            Jac[6, 6] = 1
            return Jac

        Jac[2:4, :4] = np.array([
            [np.sin(T*w)/w,(-1+np.cos(T*w))/w],
            [(1-np.cos(T*w))/w, np.sin(T*w)/w],
            [np.cos(T*w), -np.sin(T*w)],
            [np.sin(T*w), np.cos(T*w)]
        ])

        Jac[:, 6] = np.array([ 
            1/w**2*(ydot-(T*w*ydot + xdot)*np.sin(T*w)+(T*w*xdot - ydot)*np.cos(T*w)),
            1/w**2*(-xdot + (w*T*xdot- ydot)*np.sin(T*w) + (w*T*ydot + xdot)*np.cos(T*w)), 
            -T*(xdot*np.sin(T*w) + ydot * np.cos(T*w)),
            T*(xdot * np.cos(T*w) - ydot * np.sin(T*w)), 
            0, 
            0,
            1]
        )


        return Jac
    
    def Q(self, x, u, T):

        Q = np.zeros((7, 7))
        Q[:4, :4] = np.eye(4)

        Q[:2, :2] = np.eye(2)*T**3/3
        Q[2:4,2:4] = np.eye(2)*T

        Q[0:2, 2:4] = np.eye(2)*T**2/2
        Q[2:4, 0:2] = np.eye(2)*T**2/2


        Q[:4, :4] *= self.sigma_a
        Q[6, 6] = T*self.sigma_w


