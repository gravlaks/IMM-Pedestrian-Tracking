import numpy as np

class GaussState:
    def __init__(self, mean, cov):

        self.mean = np.asarray(mean, dtype=np.float32).reshape((-1, 1))
        self.cov = np.asarray(cov, dtype=np.float32)
    

    ## This function allows for tuple unpacking
    def __iter__(self):
        return iter((self.mean, self.cov))


def moments_gaussian_mixture(weights, means, covs):
    """
    6.19-6.21 in Brekke
    weights: M, 1
    means: M, n
    covs: M, n, n
    """
    mean_bar = np.average(means, weights=weights.flatten(), axis=0).reshape((-1, 1))
    cov_left = np.average(covs, weights=weights.flatten(), axis=0)

    #spread of innovations

    M = means.shape[0]
    n = means[0].shape[0]
    cov_right = np.zeros((n, n))

    for w, mean in zip(weights, means):

        mean = mean.reshape((-1, 1))
        diff = mean-mean_bar
        cov_right+=w.item()*diff@diff.T
    
    return mean_bar, cov_left+cov_right
