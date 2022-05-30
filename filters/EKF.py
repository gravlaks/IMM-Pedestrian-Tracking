import numpy as np
from measurement_models.range_bearing import RangeBearing
from utils.Gauss import GaussState
import scipy.stats
import scipy.linalg
class EKF():
    def __init__(self, dynamics_model, measurement_model):
        self.dyn_model = dynamics_model
        self.meas_model = measurement_model

    def wrap_angle(self, x):
        while x>np.pi:
            x -= np.pi*2
        while x<=-np.pi:
            x += np.pi*2
        return x
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

        if isinstance(self.meas_model, RangeBearing):
            innovation = y-y_hat
            innovation[1] = self.wrap_angle(innovation[1])
        else: 
            raise Exception( "Should use RangeBearing model")
        joseph_form = True
        if joseph_form:
            K = S@np.linalg.solve((C@S@C.T+R), C).T
            mu_nxt = mu + K@ innovation
            I = np.eye(((K@C).shape[0]))
            Sigma_nxt = (I - K @ C) @ S @ (I- K @C).T + K@ R @ K.T
        else:
            mu_nxt = mu+S@C.T@np.linalg.solve(C@S@C.T + R, y-y_hat)
            Sigma_nxt = S - S@C.T@np.linalg.solve(C@S@C.T + R, C@S)
        #if np.any(np.linalg.eigvals(Sigma_nxt)<0):
            #Sigma_nxt += np.eye(Sigma_nxt.shape[0])*1e-1
            #import pdb;pdb.set_trace()

            #raise Exception
        assert(mu.shape==mu_nxt.shape)
        if np.isnan(mu_nxt[0,0]):
            import pdb;pdb.set_trace()
        return GaussState(mu_nxt, Sigma_nxt)
    
    def loglikelihood(self, gauss_state, y):
        """
        Log likelihood of measurement.

        This function returns, for a given measurement y, 
        the result of p(y|x), where x is our posterior x

        """

        self._MLOG2PIby2 = 2* \
            np.log(2 * np.pi) / 2
        mean, cov = gauss_state
        innov = y-self.meas_model.h(mean)
        innov[1] = self.wrap_angle(innov[1])

        ## innovation covariance: 
        H = self.meas_model.H(mean)
        S = H@cov@H.T+self.meas_model.R(mean)
        #S = S + np.eye(S.shape[0])*1e-4

        # rv = multivariate_normal(np.zeros_like(y), R)

        #     L = rv.pdf(y - self.meas_model.h(x))

        # cholS = scipy.linalg.cholesky(S, lower=True)
    
        # invcholS_v = scipy.linalg.solve_triangular(cholS, innov, lower=True)
        # NISby2 = (invcholS_v ** 2).sum() / 2
        # # alternative self.NIS(...) /2 or v @ la.solve(S, v)/2
    
        # logdetSby2 = np.log(cholS.diagonal()).sum()
        # # alternative use la.slogdet(S)
    
        # ll = -(NISby2 + logdetSby2 + self._MLOG2PIby2)
        try:
            ll = scipy.stats.multivariate_normal.logpdf(innov.flatten(), cov=S, allow_singular=True)
        except:
            # sometimes it says S is singular when it isn't - what we can do is feed an argument allow_singular=True
            import pdb;pdb.set_trace()
            #ll = -1000
        #ll = max(-40, ll)

        # if ll<-1000.:
        #     import pdb;pdb.set_trace()
        return ll
        #res =  cov - cov@H.T@np.linalg.solve(S, H@cov)
