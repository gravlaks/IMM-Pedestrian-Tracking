import numpy as np
from utils.Gauss import GaussState
import scipy.stats
class iEKF():
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
        F = self.dyn_model.F(mu, u, T)

        mu_nxt = self.dyn_model.f(mu, u, T)
        Sigma_nxt = F@S@F.T + self.dyn_model.Q(mu,u, T)
    
        return GaussState(mu_nxt, Sigma_nxt)

    def update(self, gauss_state, y):
        mu, S = gauss_state
        y_hat = self.meas_model.h(mu)
        C = self.meas_model.H(mu)
        R = self.meas_model.R(mu)

        mu_i = np.copy(mu)
        eps = 1e-5
        iter_Max = 1000
        for i in range(iter_Max):
            innov = y-y_hat
            innov[1] = self.wrap_angle(innov[1])
            K = S@np.linalg.solve((C@S@C.T+R), C).T
            mu_nxt = mu + K@innov + K @C@(mu_i-mu)
            
            if np.linalg.norm(mu_nxt-mu_i)<eps:
                break
            y_hat = self.meas_model.h(mu_nxt)
            C = self.meas_model.H(mu_nxt)
            mu_i = mu_nxt
        I = np.eye(((S).shape[0]))
        Sigma_nxt = (I - K @ C) @ S @ (I- K @C).T + K@ R @ K.T
        #Sigma_nxt = S - S@C.T@np.linalg.solve(C@S@C.T + R, C@S)

        return GaussState(mu_nxt, Sigma_nxt)

    def loglikelihood(self, gauss_state, y):
        """
        Log likelihood of measurement.

        This function returns, for a given measurement y, 
        the result of p(y|x), where x is our posterior x

        """

 
        mean, cov = gauss_state
        innov = y-self.meas_model.h(mean)
        innov[1] = self.wrap_angle(innov[1])

        ## innovation covariance: 
        H = self.meas_model.H(mean)
        S = H@cov@H.T+self.meas_model.R(mean)
        
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
