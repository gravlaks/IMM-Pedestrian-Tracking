import numpy as np
from utils.Gauss import GaussState
import scipy.stats
class EKF():
    def __init__(self, dynamics_model, measurement_model):
        self.dyn_model = dynamics_model
        self.meas_model = measurement_model

    def predict(self, gauss_state, u, T):
        mu, S = gauss_state
        F = self.dyn_model.F(mu,u, T)

        mu_nxt = self.dyn_model.f(mu, u, T)
        Sigma_nxt = F@S@F.T + self.dyn_model.Q(mu,u, T)
        assert(mu.shape==mu_nxt.shape)

        # print(mu_nxt)

        if np.isnan(mu_nxt[0,0]):
            import pdb;pdb.set_trace()

        return GaussState(mu_nxt, Sigma_nxt)

    def update(self, gauss_state, y):
        mu, S = gauss_state
        y_hat = self.meas_model.h(mu)
        C = self.meas_model.H(mu)
        R = self.meas_model.R(mu)


        mu_nxt = mu+S@C.T@np.linalg.solve(C@S@C.T + R, y-y_hat)
        Sigma_nxt = S - S@C.T@np.linalg.solve(C@S@C.T + R, C@S)
        assert(mu.shape==mu_nxt.shape)
        return GaussState(mu_nxt, Sigma_nxt)
    
    def loglikelihood(self, gauss_state, y):
        """
        Log likelihood of measurement.

        This function returns, for a given measurement y, 
        the result of p(y|x), where x is our posterior x

        """

        mean, cov = gauss_state
        innov = y-self.meas_model.h(mean)

        ## innovation covariance: 
        H = self.meas_model.H(mean)
        S = H@cov@H.T+self.meas_model.R(mean)
        #S = S + np.eye(S.shape[0])*1e-4

        # rv = multivariate_normal(np.zeros_like(y), R)

        #     L = rv.pdf(y - self.meas_model.h(x))
        try:
            ll = scipy.stats.multivariate_normal.logpdf(innov.flatten(), cov=S, allow_singular=True)
        except:
            # sometimes it says S is singular when it isn't - what we can do is feed an argument allow_singular=True
            import pdb;pdb.set_trace()
        #ll = max(-40, ll)

        # if ll<-1000.:
        #     import pdb;pdb.set_trace()
        return ll
        #res =  cov - cov@H.T@np.linalg.solve(S, H@cov)
