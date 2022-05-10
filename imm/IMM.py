from random import gauss
import numpy as np
from utils.Gauss import GaussState, moments_gaussian_mixture
from imm.Gaussian_Mixture import GaussianMixture
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

        mu = pi.T@p
        mu = mu/np.sum(mu)

        return mu

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
            mean, cov = moments_gaussian_mixture(mu_prob, means, covs)

            mixes.append(GaussState(mean, cov))

        mixes = np.array(mixes)
        
        return mixes
    

        
    def filter_prediction(self, mixed_states, dt, u):
        """
        6.31 (part 1)
        """
        pred = []
        for state, filter in zip(mixed_states, self.filters):
            pred.append(filter.predict(state, dt, u))
        
        pred = np.array(pred)
        return pred
    
    def filter_update(self, mixed_states, y):
        
        upd = []
        for state, filter in zip(mixed_states, self.filters):
            upd.append(filter.update(state, y))

        upd = np.array(upd)
    
    def update_mode_probs(self, immstate, y):
        """
        6.32-33
        """
        mode_log_likelihood = []
        for (means, covs, weights), filter in zip(immstate, self.filters):
            mode_log_likelihood.append(filter.loglikelihood(y, GaussState(means, covs)))

        mode_log_likelihood = np.array(mode_log_likelihood)

        p = immstate.weights

        ## Denominator in 6.33
        normalization_factor = np.sum(
            p*np.exp(mode_log_likelihood)
        )

        ## Convert all terms to log
        log_norm_factor = np.log(normalization_factor)
        log_weights = np.log(p)

        return np.exp(log_weights+mode_log_likelihood-log_norm_factor)

    
    def predict(self, immstate, u, dt):
        """
        steps 1 and 2 
        """
        p = immstate.weights
        mixing_probs = self.mixing_probabilities(self.pi, p)

        mixes = self.mix_states(immstate, mixing_probs)

        gauss_pred = self.filter_prediction(mixes, dt, u)
        immstate = GaussianMixture(
            mixing_probs, gauss_pred
        )
        return immstate

    def update(self, immstate, y):

        gauss_upd = self.filter_update(immstate.gauss_states, y)
        weights_upd = self.update_mode_probs(immstate, y)

        immstate = GaussianMixture(
            weights_upd, gauss_upd
        )
        return immstate

    def take_step(self, immstate, u, dt, y):
        
        immstate = self.predict(immstate, u, dt)
        immstate = self.update(immstate, y)
        
        return immstate

    def get_estimate(self, immstate):

        means = np.array([
            gauss.mean for gauss in immstate.gauss_states
        ], dtype=np.float32)
        covs = np.array([
            gauss.covs for gauss in immstate.gauss_states
        ], dtype=np.float32)
        weights = immstate.weights

        mean, cov = moments_gaussian_mixture(weights, means, covs)
        return GaussState(mean, cov)