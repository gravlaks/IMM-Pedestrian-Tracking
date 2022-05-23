import sys
import numpy as np
import os
import pickle

from dynamics_models.CV_7dim import CV_7dim
from dynamics_models.CA_7dim import CA_7dim
from dynamics_models.CT_7dim import CT_7dim
from measurement_models.range_bearing import RangeBearing
from utils.plotting import plot_trajectory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_data(N, dt, mu0, cov0, process_noise=True, sensor_noise=True, run_model='SWITCH'):
    cv = CV_7dim(sigma=0.1)
    ca = CA_7dim(sigma=0.1)
    ct = CT_7dim(sigma_a=0.1, sigma_w=0.01)
    meas = RangeBearing(sigma=0.1)

    # x0 = np.array([1, 2, 1, 2, 1, 2]).reshape((-1, 1))
    x0 = np.random.randn(7, 1)
    xs = [x0]
    if run_model == 'CV':
        cv_model=True
    elif run_model == 'CA':
        cv_model=False
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
            if i%200==0:
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
                    xcurr[4:] = np.random.randn(3).reshape(-1, 1)

                switches.append(i)

        if cv_model:
            dyn = cv
        else:
            dyn = ca # TURN THIS BW CT, CA

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

    return np.array(xs), np.array(zs)


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
    # mu0 = np.zeros((7, 1))
    # cov0 = np.eye(7)
    # xs, zs = generate_data(1000, 0.1, mu0, cov0, process_noise=True, sensor_noise=True, run_model='CA')
    # # xs, zs = get_data(0, process_noise=False, sensor_noise=False)
    # # import pdb;pdb.set_trace()
    # plot_trajectory(xs, zs)
    # np.save('x.npy', xs)
    # np.save('z.npy', zs)
    xs, zs = get_data(0, RangeOnly(sigma=0.1), False, True)
    plot_trajectory(xs, zs)
