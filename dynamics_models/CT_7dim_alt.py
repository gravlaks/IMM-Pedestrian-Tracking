from tkinter import W
import numpy as np
from dataclasses import dataclass

@dataclass
class CT_7dim_alt():
    """
    n is dimension of state space
    sigma is noise 
    """
    sigma_a: float
    sigma_w: float
    n: int=2

    def f(self, x, u, T):
        w = x[-1]
        if w == 0:
            w = 0.00001
        Transition = np.zeros((7, 7))
        Transition[:4, :4] = np.eye(4)
        Transition[:4, 2:4] = np.array([
            [np.sin(T*w)/w,(-1+np.cos(T*w))/w],
            [(1-np.cos(T*w))/w, np.sin(T*w)/w],
            [np.cos(T*w), -np.sin(T*w)],
            [np.sin(T*w), np.cos(T*w)]
        ]).reshape((-1, 2))
        Transition[6, 6] = 1
        return Transition@x

    def F(self, x, u, T):
        w = x[-1]
        Jac = np.zeros((7, 7))
        Jac[:4, :4] = np.eye(4)
        xdot, ydot = x[2], x[3]
        if w == 0:
            Jac[0:2, 2:4] = np.eye(2)*T
            Jac[:4, 6] = np.array([-T**2*ydot/2,T**2*xdot/2, -T*ydot, T*xdot]).flatten()
            Jac[6, 6] = 1
            return Jac

        Jac[:4, 2:4] = np.array([
            [np.sin(T*w)/w,(-1+np.cos(T*w))/w],
            [(1-np.cos(T*w))/w, np.sin(T*w)/w],
            [np.cos(T*w), -np.sin(T*w)],
            [np.sin(T*w), np.cos(T*w)]
        ]).reshape((-1, 2))

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
        sa2 = self.sigma_a**2
        w = x[-1]
        if w == 0:
            w = 1e-7

        Q = np.zeros((7, 7))
        """

        2*sa**2*(T*w - sin(T*w))/w**3      0      sa**2*(1 - cos(T*w))/w**2      sa**2*(T*w - sin(T*w))/w**2      
0      2*sa**2*(T*w - sin(T*w))/w**3      sa**2*(-T*w + sin(T*w))/w**2      sa**2*(1 - cos(T*w))/w**2      
sa**2*(1 - cos(T*w))/w**2      sa**2*(-T*w + sin(T*w))/w**2      T*sa**2      0      
sa**2*(T*w - sin(T*w))/w**2      sa**2*(1 - cos(T*w))/w**2      0      T*sa**2

        """

        Q[0, 0] = 2*sa2*(T*w-np.sin(T*w))/w**3
        Q[0, 2] = sa2*(1-np.cos(T*w))/w**2
        Q[0, 3] = sa2*(T*w-np.sin(T*w))/w**2
        Q[1, 1] = 2*sa2*(T*w-np.sin(T*w))/w**3
        Q[1, 2] = sa2*(-T*w + np.sin(T*w))/w**2
        Q[1, 3] = sa2*(1-np.cos(T*w))/w**2

        Q[2, 0] = sa2*(1-np.cos(T*w))/w**2
        Q[2, 1] = sa2*(-T*w + np.sin(T*w))/w**2
        Q[2, 3] = T*sa2

        Q[3, 0] = sa2*(T*w - np.sin(T*w))/w**2
        Q[3, 1] = sa2*(1-np.cos(T*w))/w**2
        Q[3, 3] = T*sa2

       

        Q[6, 6] = T*self.sigma_w**2


        return Q