
import matplotlib.pyplot as plt
import numpy as np
def plot(x_GT, gauss_est):
    mus = []
    for gauss in gauss_est: 
        mu, cov = gauss
        mus.append(mu)

    mus = np.array(mus)
    x_GT = np.array(x_GT)
    plt.plot(mus[:, 0], label="px est")
    plt.plot(mus[:, 1], label="py est")
    plt.plot(mus[:, 2], label="theta est")
    plt.plot(x_GT[:, 0], label="px GT")
    plt.plot(x_GT[:, 1], label="py GT")
    plt.plot(x_GT[:, 2], label="theta GT")
    plt.legend()
    plt.show()