from random import gauss
import numpy as np
from utils.Gauss import GaussState, moments_gaussian_mixture
from imm.Gaussian_Mixture import GaussianMixture
from dynamics_models.CA_7dim import CA_7dim
from dynamics_models.CV_7dim import CV_7dim
from dynamics_models.CT_7dim import CT_7dim

class IMM():
    def __init__(self, filters, pi):
        self.filters = filters
        self.pi = pi

    def mixing_probabilities(self, pi, p):
        """
        6.27 Brekke
        pi: M, M
        p = M, 1
        """

        mu = pi*p
        mu = mu/(pi.T@p)

        return mu, pi.T@p

    def get_dyn_types(self):
        dyn_models = [fil.dyn_model for fil in self.filters]
        dyn_type = []

        for dyn in dyn_models:
            if isinstance(dyn, CA_7dim):
                dyn_type.append('CA')
            elif isinstance(dyn, CV_7dim):
                dyn_type.append('CV')
            elif isinstance(dyn, CT_7dim):
                dyn_type.append('CT')

        return dyn_type

    def mix_states(self, immstate, mu_probs):
        """
        6.28-30
        immstate: ImmState, contains M Gaussians (each conditioned on a certain mode)
        mu_probs: M, M    mu_ij = mu_sk, sk-1
        """
        means = []
        covs = []
        for g_s in immstate.gauss_states:
            mean, cov = g_s
            means.append(mean)
            covs.append(cov)
        means, covs = np.array(means), np.array(covs)

        mixes = []
        for mu_prob in mu_probs:
            mean, cov = moments_gaussian_mixture(mu_prob.reshape((-1, 1)), means, covs, self.get_dyn_types())

            mixes.append(GaussState(mean, cov))

        mixes = np.array(mixes)
        
        return mixes
    
    def filter_prediction(self, mixed_states, dt, u):
        """
        6.31 (part 1)
        """
        pred = []
        for state, filter in zip(mixed_states, self.filters):
            pred.append(filter.predict(state, u, dt))
            # print(filter.predict(state, u, dt).mean)
        
        pred = np.array(pred)
        return pred
    
    def filter_update(self, mixed_states, y):
        
        upd = []
        for state, filter_ in zip(mixed_states, self.filters):
            upd.append(filter_.update(state, y))

        upd = np.array(upd)
        return upd
    
    def update_mode_probs(self, immstate, y):
        """
        6.32-33
        """
        mode_log_likelihood = []
        for gauss_mixt, filter_ in zip(immstate.gauss_states, self.filters):
            means, covs = gauss_mixt
            mode_log_likelihood.append(filter_.loglikelihood(GaussState(means, covs), y))

        mode_log_likelihood = np.array(mode_log_likelihood).reshape((-1, 1))

        mode_log_likelihood[mode_log_likelihood<-500] = -500

        p = immstate.weights

        ## Denominator in 6.33
        normalization_factor = np.sum(
            p*np.exp(mode_log_likelihood)
        ) 
        # log_norm_factor = np.log(p)+mode_log_likelihood #sometimes this is 0 when the likelihoods are super low- need to make sure this doesn't happen
        if normalization_factor == 0.0 or np.isnan(normalization_factor):
            # import pdb;pdb.set_trace()
            print('normalization factor 0 or nan')
            normalization_factor = 1e-200
            # weights = np.ones_like(immstate.weights)/len(immstate.weights)
            # weights = -p*1/mode_log_likelihood
            # return p

        ## Convert all terms to log
        log_norm_factor = np.log(normalization_factor)
        log_weights = np.log(p)

        weights = np.exp(log_weights+mode_log_likelihood-log_norm_factor)

        if weights[0] > 1000000000000000000:
            import pdb;pdb.set_trace()

        # print(normalization_factor)
        # print(weights)

        return weights

    
    def predict(self, immstate, u, T):
        """
        steps 1 and 2 
        """
        p = immstate.weights
        mixing_probs, weights = self.mixing_probabilities(self.pi, p)
    
            
        mixes = self.mix_states(immstate, mixing_probs)

        gauss_pred = self.filter_prediction(mixes, T, u)
        # print(gauss_pred[0].mean)
        # print(gauss_pred[1].mean)

        # weights = np.maximum(1e-10,weights )
        immstate = GaussianMixture(
            weights, gauss_pred
        )
        return immstate

    def update(self, immstate, y):
        
        gauss_upd = self.filter_update(immstate.gauss_states, y)
        # if np.any(np.diag(gauss_upd[0].cov)<0) or np.any(np.diag(gauss_upd[1].cov)<0):
        #     import pdb;pdb.set_trace()
        weights = self.update_mode_probs(immstate, y)   
        # weights = np.maximum(1e-10, weights)
        # if np.any(weights<1e-9):
        #     print(weights)

        immstate = GaussianMixture(
            weights, gauss_upd
        )

        gauss, _ = self.get_estimate(immstate)
        if np.isnan(gauss.mean[0,0]):
            gauss, _ = self.get_estimate(immstate)
            import pdb;pdb.set_trace()
        return immstate

    def take_step(self, immstate, u, dt, y):
        
        
        immstate = self.predict(immstate, u, dt)
        immstate = self.update(immstate, y)
        
        return immstate

    def get_estimate(self, immstate):

        means = np.array([
            gauss.mean.flatten() for gauss in immstate.gauss_states
        ], dtype=np.float32)
        covs = np.array([
            gauss.cov for gauss in immstate.gauss_states
        ], dtype=np.float32).squeeze()
        weights = immstate.weights

        mean, cov = moments_gaussian_mixture(weights, means, covs, self.get_dyn_types())
        return GaussState(mean, cov), weights