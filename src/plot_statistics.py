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

    for j in range(3):

        fig, ax = plt.subplots(2, 1)
        fig.set_tight_layout(True)

        if meas_states:
            #TODO - associating measurements with certain states so we can plot measurements only for measured quantities
            raise NotImplementedError

        for k, i in enumerate(range(2*j, 2*j+2)):
            ax[k].fill_between(times, mu[:, i]-1.96*np.sqrt(Sigma[:, i, i]), mu[:, i]+1.96*np.sqrt(Sigma[:, i, i]), color='b', alpha=0.1, label=f'95% Confidence')
            # if len(y[0]) > i:
            #     ax[i].scatter(times, y[:, i], c='r', s=8, label='meas')
            ax[k].plot(times, x[:, i], 'k', label='Truth')
            ax[k].plot(times, mu[:, i], 'r', label='Mean Estimate')
            ax[k].legend()
            ax[k].set_title(f'{savestr} {titles[i]} Statistics') #  for {int(np.ceil(max_t))} seconds
            ax[k].set_ylabel(f'{titles[i]}')

            if k==1:
                ax[k].set_xlabel('Time')
        if savestr:
            plt.savefig(f'../plots/examples/{savestr}_{j}.png')


    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    i = -1

    ax.fill_between(times, mu[:, i]-1.96*np.sqrt(Sigma[:, i, i]), mu[:, i]+1.96*np.sqrt(Sigma[:, i, i]), color='b', alpha=0.1, label=f'95% Confidence')
    # if len(y[0]) > i:
    #     ax[i].scatter(times, y[:, i], c='r', s=8, label='meas')
    ax.plot(times, x[:, i], 'k', label='Truth')
    ax.plot(times, mu[:, i], 'r', label='Mean estimate')
    ax.legend()
    ax.set_title(f'{savestr} {titles[i]} Statistics')
    ax.set_ylabel(f'{titles[i]}')
    ax.set_xlabel('Time')
    if savestr:
        plt.savefig(f'../plots/examples/{savestr}_4.png')

    fig, ax = plt.subplots(2,2)
    fig.set_tight_layout(True)

    pos_err = np.linalg.norm(x[:,:2] - mu[:,:2], axis=1)
    vel_err = np.linalg.norm(x[:,:4] - mu[:,:4], axis=1)
    acc_err = np.linalg.norm(x[:,:6] - mu[:,:6], axis=1)
    w_err = np.abs(x[:,6] - mu[:,6])

    i = -1

    ax[0,0].plot(times, pos_err, 'k', label='pos err')
    ax[0,1].plot(times, vel_err, 'k', label='vel err')
    ax[1,0].plot(times, vel_err, 'k', label='acc err')
    ax[1,1].plot(times, w_err, 'k', label='turn rate err')
    for i in range(2):
        for j in range(2):
            ax[i,j].legend()
    plt.title(f'Tracking Errors vs Time')
    ax[0,0].set_ylabel(f'Magnitude State Error')
    ax[1,0].set_ylabel(f'Magnitude State Error')
    ax[1,0].set_xlabel(f'Time')
    ax[1,1].set_xlabel(f'Time')
    if savestr:
        plt.savefig(f'../plots/examples/{savestr}_errors.png')

    