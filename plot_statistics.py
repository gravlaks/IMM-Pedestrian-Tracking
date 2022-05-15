import numpy as np
from matplotlib import pyplot as plt

def plot_statistics(times, mu, Sigma, x, y, titles, savestr, meas_states=None):
    ''' times: array of time steps
    mu: posterior mean
    Sigma: posterior Cov
    x: truth trajectory (of same dim as state)
    y: measurements
    titles: list of strings to name states (of same dim as state)
    '''

    n = len(mu[0])

    max_t = times[-1]

    fig, ax = plt.subplots(n, 1)
    fig.set_tight_layout(True)

    if meas_states:
        #TODO - associating measurements with certain states so we can plot measurements only for measured quantities
        raise NotImplementedError

    for i in range(n):
        ax[i].fill_between(times, mu[:, i]-1.96*np.sqrt(Sigma[:, i, i]), mu[:, i]+1.96*np.sqrt(Sigma[:, i, i]), color='b', alpha=0.1, label='95%% Confidence')
        # if len(y[0]) > i:
        #     ax[i].scatter(times, y[:, i], c='r', s=8, label='meas')
        ax[i].plot(times, x[:, i], 'k', label='truth')
        ax[i].plot(times, mu[:, i], 'r', label='mean estimate')
        ax[i].legend()
        ax[i].set_title(f'{savestr} {titles[i]} statistics for {int(np.ceil(max_t))} seconds')
        ax[i].set_ylabel(f'{titles[i]}')

        if i==n-1:

            ax[i].set_xlabel('time')
    if savestr:
        plt.savefig(f'{savestr}.png')