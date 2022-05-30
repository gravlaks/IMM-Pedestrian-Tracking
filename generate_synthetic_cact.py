import sys
import numpy as np
import os
import pickle

from dynamics_models.CV import CV
from dynamics_models.CA import CA
from dynamics_models.CT import CT
from measurement_models.range_bearing import RangeBearing
from utils.plotting import plot_trajectory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_data(N, dt, mu0, cov0, process_noise=True, sensor_noise=True, run_model='SWITCH'):
    cv = CV(sigma=0.1)
    ca = CA(sigma=0.05)
    ct = CT(sigma_a=0.01, sigma_w=0.0001)
    meas = RangeBearing(sigma_r=0.1, sigma_th=0.001)

    # x0 = np.array([1, 2, 1, 2, 1, 2]).reshape((-1, 1))
    x0 = np.random.randn(7, 1)
    xs = [x0]
    if run_model == 'CV':
        cv_model=True
    elif run_model == 'CA':
        cv_model=False
    elif run_model == 'CT':
        cv_model = False
    else:
        cv_model=True
    z0 = np.array([0, 0]).reshape((-1, 1))
    zs = [z0]
    u = np.array([0, 0])
    switches = []
    
    for i in range(N-1):
        xcurr = xs[-1]

        #Change model 
        if run_model=='SWITCH':
            if i%50==0:
                # if cv_model:
                #     xcurr[2:4] = np.zeros((2, 1))

                #     # if i%2:
                #     #     xcurr[4:] = 1e-3*np.array([1, -5]).reshape((-1, 1))
                #     # if i%2==0:
                #     #     xcurr[4:] = 1e-3*np.array([1, 5]).reshape((-1, 1))

                #     xcurr[4:] = 1e-3*np.array([1, 5]).reshape((-1, 1))

                # else:
                #     xcurr[4:] = np.zeros((2,)).reshape((-1, 1))
                #     xcurr[2:4] = -xcurr[2:4]
                cv_model = not cv_model

                if not cv_model:
                    xcurr[4:6] = 0.001*np.random.randn(2).reshape(-1, 1)
                    xcurr[6] = np.random.randn(1).reshape(-1, 1)
                switches.append(i)

        if cv_model:
            dyn = ca
        else:
            dyn = ct # TURN THIS BW CT, CA

        # print(dyn)

        if process_noise:
            #if isinstance(dyn, CA):
            #    print(dyn.Q(xcurr, u=None, T=dt))
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


def get_data(traj_num, sensor_model, process_noise=True, sensor_noise=True):
    # Get trajectory from pedestrian trajectory dataset
    dict_file = open(os.path.join(SCRIPT_DIR,'ped_data.pkl'),"rb")
    trajectories = pickle.load(dict_file)
    dict_file.close()

    xs = trajectories[str(traj_num)]

    N, _ = xs.shape

    zs = np.array([sensor_model.h(xs[i,:]) for i in range(N)])
    zs = zs[:,:,0]

    _, M = zs.shape

    if process_noise:
        Q = np.diag([0.005, 0.005, 0.001, 0.001, 0.0005, 0.0005])
        xs += np.random.multivariate_normal(np.zeros(7), Q, N)

    if sensor_noise:
        zs += np.random.multivariate_normal(np.zeros(M), sensor_model.R(xs), N)
    
    zs = zs[:,:,np.newaxis]

    return xs, zs

if __name__=='__main__':
    from measurement_models.range_only import RangeOnly
    mu0 = np.zeros((7, 1))
    cov0 = np.eye(7)
    xs, zs, switches = generate_data(100, 0.1, mu0, cov0, process_noise=True, sensor_noise=True, run_model='SWITCH')
    # # xs, zs = get_data(0, process_noise=False, sensor_noise=False)
    # # import pdb;pdb.set_trace()
    # plot_trajectory(xs, zs)
    np.save('x.npy', xs)
    np.save('z.npy', zs)
    np.save('switches.npy', switches)
    # xs, zs = get_data(0, RangeOnly(sigma=0.1), False, True)
    plot_trajectory(xs, zs)
