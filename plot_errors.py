import numpy as np
from matplotlib import pyplot as plt

def plot_errors(times, mus, Sigmas, x, y, titles, savestr, meas_states=None, switches=None):
    ''' times: array of time steps
    mu: posterior mean
    Sigma: posterior Cov
    x: truth trajectory (of same dim as state)
    y: measurements
    titles: list of strings to name states (of same dim as state)
    '''

    max_t = times[-1]

    fig, ax = plt.subplots(2,2)
    fig.set_tight_layout(True)

    for idx, flt in enumerate(mus.keys()):

        pos_err = np.linalg.norm(x[:,:2] - mus[flt][:,:2], axis=1)**2
        vel_err = np.linalg.norm(x[:,:4] - mus[flt][:,:4], axis=1)**2
        acc_err = np.linalg.norm(x[:,:6] - mus[flt][:,:6], axis=1)**2
        w_err   = np.abs(x[:,6] - mus[flt][:,6])**2

        max_errors = [np.max(pos_err), np.max(vel_err), np.max(acc_err), np.max(w_err)]

        pos_err = np.convolve(pos_err, 1/5*np.ones(5), mode='same')
        vel_err = np.convolve(vel_err, 1/5*np.ones(5), mode='same')
        acc_err = np.convolve(acc_err, 1/5*np.ones(5), mode='same')
        w_err = np.convolve(w_err, 1/5*np.ones(5), mode='same')

        i = -1

        ax[0,0].plot(times, pos_err, label=f'{titles[idx]}, MSE: {np.mean(pos_err):0.3f}')
        ax[0,1].plot(times, vel_err, label=f'{titles[idx]}, MSE: {np.mean(vel_err):0.3f}')
        ax[1,0].plot(times, acc_err, label=f'{titles[idx]}, MSE: {np.mean(acc_err):0.3f}')
        ax[1,1].plot(times, w_err, label=f'{titles[idx]}, MSE: {np.mean(w_err):0.3f}')
        for i in range(2):
            for j in range(2):
                ax[i,j].legend()
                if switches is not None:
                    for k in range(len(switches)):
                        ax[i,j].plot([switches[k], switches[k]], [0, max_errors[2*i+j]], 'k:')

        ax[0,0].set_title(f'Position Squared Error')
        ax[0,1].set_title(f'Velocity Squared Error')
        ax[1,0].set_title(f'Acceleration Squared Error')
        ax[1,1].set_title(f'Turn Rate Squared Error')

        ax[0,0].set_ylabel(f'Magnitude State Squared Error')
        ax[1,0].set_ylabel(f'Magnitude State Squared Error')
        ax[1,0].set_xlabel(f'Time')
        ax[1,1].set_xlabel(f'Time')

        ax[0,0].set_ylim([0, 0.22])
        ax[0,1].set_ylim([0, 1.75])
        ax[1,0].set_ylim([0, 2.5])
        ax[1,1].set_ylim([0.0, 1.2])

    if savestr:
        plt.savefig(f'{savestr}_errors.png')

    