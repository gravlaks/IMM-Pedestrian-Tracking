import numpy as np

class GaussState:
    def __init__(self, mean, cov):

        self.mean = np.asarray(mean, dtype=np.float32).reshape((-1, 1))
        self.cov = np.asarray(cov, dtype=np.float32)
    

    ## This function allows for tuple unpacking
    def __iter__(self):
        return iter((self.mean, self.cov))

def adjust_acc_averages(idx_ct, idx_ca, means, covs, weights):
    w = []
    means_acc = []
    covs_acc = []

    weights = np.copy(weights.flatten())

    for ict in idx_ct:
        w.append(weights[ict])
        means_acc.append(means[ict, 4:])
        covs_acc.append(covs[ict, 4:, 4:])

    for ica in idx_ca:
        w.append(weights[ica])
        means_acc.append(means[ica, 4:])
        covs_acc.append(covs[ica, 4:, 4:])

    weights_acc = np.array(w/np.sum(w))
    acc_mean = np.array(means_acc)
    acc_cov = np.array(covs_acc)

    mean_bar_acc = np.average(acc_mean, weights=weights_acc, axis=0).reshape(-1,1)
    cov_left_acc = np.average(acc_cov, weights=weights_acc, axis=0)

    return mean_bar_acc, cov_left_acc


def moments_gaussian_mixture(weights, means, covs, dyn_type):
    """
    6.19-6.21 in Brekke
    weights: M, 1
    means: M, n
    covs: M, n, n
    """

    weights_flat = weights.flatten()

    mean_bar = np.average(means, weights=weights.flatten(), axis=0).reshape((-1, 1))
    cov_left = np.average(covs, weights=weights.flatten(), axis=0)

    idx_ct = [i for i in range(len(dyn_type)) if dyn_type[i]=='CT']
    idx_ca = [i for i in range(len(dyn_type)) if dyn_type[i]=='CA']

    # if len(idx_ct) > 0 and len(idx_ca) ==0:
    #     mean_bar[4:, :] = means[idx_ct, 4:].T
    #     cov_left[4:, 4:] = covs[idx_ct, 4:, 4:]
    # elif len(idx_ca) > 0 and len(idx_ct) ==0:
    #     mean_bar[4:, :] = means[idx_ca, 4:].T
    #     cov_left[4:, 4:] = covs[idx_ca, 4:, 4:]
    # elif len(idx_ct) == 0 and len(idx_ca) == 0:
    #     pass
    # else:
    #     # print('averaging accelerations and omega for CT and CA together not yet implemented')
    #     w_ct = weights_flat[idx_ct]/(weights_flat[idx_ct] + weights_flat[idx_ca])
    #     w_ca = weights_flat[idx_ca]/(weights_flat[idx_ct] + weights_flat[idx_ca])
    #     weights_acc = np.array([w_ct, w_ca]).flatten()
    #     acc_mean = np.concatenate((means[idx_ct, 4:], means[idx_ca, 4:]))
    #     acc_cov = np.concatenate((covs[idx_ct, 4:, 4:], covs[idx_ca, 4:, 4:]))
    #     mean_bar[4:, :] = np.average(acc_mean, weights=weights_acc, axis=0).reshape(-1,1)
    #     cov_left[4:, 4:] = np.average(acc_cov, weights=weights_acc, axis=0)
    #     # pass

    if len(idx_ct) > 0 or len(idx_ca) > 0:
        mean_bar[4:, :], cov_left[4:, 4:] = adjust_acc_averages(idx_ct, idx_ca, means, covs, weights)

    #spread of innovations

    M = means.shape[0]
    n = means[0].shape[0]
    cov_right = np.zeros((n, n))

    for w, mean in zip(weights, means):

        mean = mean.reshape((-1, 1))
        diff = mean-mean_bar
        cov_right+=w.item()*diff@diff.T
    
    return mean_bar, cov_left+cov_right
