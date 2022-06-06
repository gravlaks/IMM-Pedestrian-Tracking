import sys
import numpy as np
import os
import pickle

from dynamics_models.CV import CV
from dynamics_models.CA import CA
from dynamics_models.CT import CT
from measurement_models.range_bearing import RangeBearing
from utils.plotting import plot_trajectory
from itertools import cycle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_data(N, dt, mu0, cov0, process_noise=True, sensor_noise=True, run_model='SWITCH'):
    cv = CV(sigma=0.01)
    ca = CA(sigma=0.05)
    ct = CT(sigma_a=0.01, sigma_w=0.0001)
    meas = RangeBearing(sigma_r=0.1, sigma_th=0.001)

    model_it = cycle([cv, ct, ca])

    # x0 = np.array([1, 2, 1, 2, 1, 2]).reshape((-1, 1))
    x0 = np.random.randn(7, 1)
    xs = [x0]
    if run_model == 'CV':
        dyn=cv
    elif run_model == 'CA':
        dyn=ca
    elif run_model == 'CT':
        dyn=ct
    z0 = np.array([0, 0]).reshape((-1, 1))
    zs = [z0]
    u = np.array([0, 0])
    switches = []
    
    for i in range(N-1):
        xcurr = xs[-1]

        #Change model 
        if run_model=='SWITCH':
            if i%200==0:

                dyn = next(model_it)

                if isinstance(dyn, CA):
                    xcurr[4:6] = 1e-1*np.random.randn(2).reshape(-1, 1)
                elif isinstance(dyn, CT):
                    xcurr[6] = np.random.randn(1).reshape(-1, 1)

                switches.append(i)

                print(dyn)

        if process_noise:
            xcurr = np.random.multivariate_normal(dyn.f(xcurr, u=None, T=dt).flatten(), dyn.Q(xcurr, u=None, T=dt)).reshape((-1, 1))
        else:
            xcurr = dyn.f(xcurr, u=None, T=dt)
        if sensor_noise:
            z = np.random.multivariate_normal(meas.h(xcurr).flatten(), meas.R(xcurr)).reshape((-1, 1))
        else:
            z = meas.h(xcurr)

        # print(xcurr)
        zs.append(z)
        xs.append(xcurr)

    return np.array(xs), np.array(zs), switches

if __name__=='__main__':
    mu0 = np.zeros((7, 1))
    cov0 = np.eye(7)
    xs, zs, switches = generate_data(1000, 0.1, mu0, cov0, process_noise=True, sensor_noise=True, run_model='SWITCH')
    np.save('../data/trajectories/x.npy', xs)
    np.save('../data/trajectories/z.npy', zs)
    np.save('../data/trajectories/switches.npy', switches)
    plot_trajectory(xs, zs)
    # plot_trajectory(xs, zs)
