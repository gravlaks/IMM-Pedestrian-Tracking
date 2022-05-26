from nbformat import read
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from dynamics_models.CV_7dim import CV_7dim
from dynamics_models.CA_7dim import CA_7dim
from dynamics_models.CT_7dim import CT_7dim
from measurement_models.range_only import RangeOnly
from measurement_models.range_bearing import RangeBearing
# from measurement_models.range_bearing import RangeOnly
from filters.EKF import EKF
from filters.UKF import UKF
from filters.iEKF import iEKF
from utils.Gauss import GaussState
from read_data import read_data
from imm.Gaussian_Mixture import GaussianMixture
from imm.IMM import IMM
from generate_synthetic import generate_data
from get_ped_data import get_data
from plot_statistics import plot_statistics

def run_filter(filt, 
               gaussStates:list,
               X:np.ndarray, 
               Z:np.ndarray, 
               model_weights=[],
               immstate=None,
               filt_type='imm',
               dt=1/30,
               visualize=False): 

    N = len(X)

    filt_x = np.zeros_like(X)
    timer_ = 0

    for i in range(1, N):

        if i % 5 == 0:
            z = Z[i]
            meas_flag = True
        else:
            meas_flag = False

        start_time = time.time()
        if filt_type == 'imm':
            immstate       = filt.predict(immstate,u=None, T=dt)
            gauss, weights = filt.get_estimate(immstate)

            if meas_flag:
                immstate       = filt.update(immstate, z)
                gauss, weights = filt.get_estimate(immstate)

            gaussStates.append(gauss)
            model_weights.append(weights.flatten())
        else:
            gauss       = filt.predict(gaussStates[i-1],u=None, T=dt)

            if meas_flag:
                gauss       = filt.update(gauss, z)

            gaussStates.append(gauss)
        timer_ += time.time() - start_time

    mus    = []
    Sigmas = []

    model_weights = np.array(model_weights)

    for gauss in gaussStates:
        mus.append(gauss.mean)
        Sigmas.append(gauss.cov)

    mus    = np.array(mus)
    Sigmas = np.array(Sigmas)

    if visualize:
        plt.plot(X[:, 0], X[:, 1], label="GT")
        plt.plot(mus[:, 0], mus[:, 1], label="Estimate")
        for i in range(N):
            T = f"{(i*dt):.0f}"
            if i % (N//10) == 0:
                plt.annotate(T, (mus[i, 0], mus[i, 1]))

        plt.legend()
        plt.title('Filter vs. True Traj')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')

        plt.figure()

        if filt_type == 'imm':

            for i in range(model_weights.shape[1]):

                plt.plot(model_weights[:,i],
                    label= filt.filters[i].dyn_model.__class__.__name__
                )
            plt.legend()
            plt.title('Model Weights')
            plt.xlabel('Time')
            plt.ylabel('Weight')
            plt.grid(True)

            times = dt*np.ones(N)
            times = np.cumsum(times)

            plot_statistics(times, 
                            mus.squeeze(), 
                            Sigmas.squeeze(), 
                            X.squeeze(), 
                            Z.squeeze(), 
                            ['px', 'py', 'vx', 'vy', 'ax', 'ay', 'w'],
                            filt_type, 
                            meas_states=None)

        plt.show()

    return np.linalg.norm(X[:,:2]-mus[:,:2,0])**2, timer_

def _setup_imm(x0, 
               dyn_models,
               sensor_models,
               base_filt='ekf',
               n_states=7, 
               p=0.95):

    if base_filt == 'ekf':
        base = lambda x,y: EKF(x,y)
    elif base_filt == 'ukf':
        base = lambda x,y: UKF(x,y)
    elif  base_filt == 'iekf':
        base = lambda x,y: iEKF(x,y)

    filters = []
    mu_init = []
    cov_init = []
    for i in range(len(dyn_models)):
        filters.append(base(dyn_models[i], sensor_models[i]))
        mu_init.append(np.append(x0,0))
        cov_init.append(np.eye(n_states))

    n_filt = len(filters)

    # uniform weights
    init_weights = np.ones((n_filt, 1))/n_filt

    init_states = [GaussState(mu_init[i], cov_init[i]) for i in range(n_filt)]

    immstate = GaussianMixture(
        init_weights, init_states
    )

    # transition probabilities
    p_switch = ((1-p)/(n_filt-1))
    PI = np.eye(n_filt)
    PI *= (p - p_switch)
    PI += p_switch

    # init IMM
    imm = IMM(filters, PI)

    # get initial gauss mixture state
    gauss0, _       = imm.get_estimate(immstate)
    gaussStates     = [gauss0]

    return imm, immstate, gaussStates

def _setup_filter(x0, dyn_model, meas_model, base_filt='ekf'):

    if base_filt == 'ekf':
        base = lambda x,y: EKF(x,y)
    elif base_filt == 'ukf':
        base = lambda x,y: UKF(x,y)
    elif  base_filt == 'iekf':
        base = lambda x,y: iEKF(x,y)

    filt = base(dyn_model, meas_model)

    mu_init  = np.append(x0,0)
    cov_init = np.eye(n_states)

    init_states = GaussState(mu_init, cov_init)

    gaussStates     = [init_states]
 
    return filt, None, gaussStates

if __name__ == "__main__":
    n_states = 7
    T = 1/30
    sigma_q = 0.1
    sigma_r = 0.1
    sigma_t = 0.02
    sigma_a = 0.1
    sigma_w = 0.01

    sensor_models = [RangeBearing(sigma_r, sigma_t, state_dim=7),
                     RangeBearing(sigma_r, sigma_t, state_dim=7),
                     RangeBearing(sigma_r, sigma_t, state_dim=7)]

    dyn_models = [CV_7dim(sigma_q),
                  CA_7dim(sigma_q),
                  CT_7dim(sigma_a=sigma_a, sigma_w=sigma_w)]
    
    n_runs = 318
    mse_all = np.zeros(n_runs)
    timer_ = np.zeros(n_runs)
    N_ = np.zeros(n_runs)
    for i in range(n_runs):
        X, Z = get_data(i, sensor_models[0], process_noise=False, sensor_noise=True)

        N_[i] = X.shape[0]

        filt, state, gauss = _setup_imm(X[0,:], dyn_models, sensor_models)

        # filt, state, gauss = _setup_filter(X[0,:], dyn_models[0], sensor_models[0])

        mse_all[i], timer_[i] = run_filter(filt, 
                        gaussStates=gauss, 
                        X=X, 
                        Z=Z, 
                        model_weights=[],
                        immstate=state,
                        filt_type='imm',
                        dt=1/30,
                        visualize=False) 

    print('Avg. Time over {} Runs: {}'.format(n_runs, timer_.mean()))
    print('S.D. Time over {} Runs: {}'.format(n_runs, timer_.std()))
    print('Avg. Time per iter: {}'.format((timer_/N_).mean()))
    print('S.D. Time per iter: {}'.format((timer_/N_).std()))
    print('MSE over {} Runs: {}'.format(n_runs, mse_all.mean()))
    print('SDE over {} Runs: {}'.format(n_runs, mse_all.std()))

    sorted_mse = np.sort(mse_all)
    _ = plt.hist(sorted_mse, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

