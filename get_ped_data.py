import sys
import numpy as np
import os
import pickle

from dynamics_models.CV import CV
from dynamics_models.CA import CA
from dynamics_models.CT import CT
from measurement_models.range_bearing import RangeBearing
from measurement_models.range_only import RangeOnly
from utils.plotting import plot_trajectory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

if __name__ == "__main__":
    xs, zs = get_data(0, RangeOnly(sigma=0.1), False, True)
    plot_trajectory(xs,zs)

