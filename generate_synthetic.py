import sys
import numpy as np

from dynamics_models.CV_inc import CV
from dynamics_models.CA import CA
from measurement_models.range_bearing import RangeBearing
from utils.plotting import plot_trajectory

def generate_data(N, dt, mu0, cov0):
    cv = CV(sigma=0.1)
    ca = CA(sigma=0.1)
    meas = RangeBearing(sigma=0.1)

    x0 = np.array([1, 1, 2, 2, 0, 0]).reshape((-1, 1))
    xs = [x0]
    cv_model = True
    zs = []
    
    for i in range(N-1):
        xcurr = xs[-1]

        #Change model 
        if i%200==0:
            if cv_model:
                xcurr[2:4] = np.zeros((2, 1))

                if i%2:
                    xcurr[4:] = 1e-3*np.array([1, -5]).reshape((-1, 1))
                if i%2==0:
                    xcurr[4:] = 1e-3*np.array([1, 5]).reshape((-1, 1))

            else:
                xcurr[4:] = np.zeros((2,)).reshape((-1, 1))
                xcurr[2:4] = -xcurr[2:4]
            cv_model= not cv_model
        

        if cv_model:
            dyn = cv

        else:
            dyn = ca
        xcurr = np.random.multivariate_normal(dyn.f(xcurr, u=None, T=dt).flatten(), dyn.Q(xcurr, u=None, T=dt)).reshape((-1, 1))
        z = np.random.multivariate_normal(meas.h(xcurr).flatten(), meas.R(xcurr)).reshape((-1, 1))
        zs.append(z)
        xs.append(xcurr)
    return np.array(xs), np.array(zs)
if __name__=='__main__':
    mu0 = np.zeros((6, 1))
    cov0 = np.eye(6)
    xs, zs = generate_data(1000, 0.1, mu0, cov0)
    plot_trajectory(xs, zs)