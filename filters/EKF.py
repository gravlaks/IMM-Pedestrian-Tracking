import numpy as np

class GaussState:
    def __init__(self, mean, cov):

        self.mean = np.asarray(mean, dtype=np.float32).reshape((-1, 1))
        self.cov = np.asarray(cov, dtype=np.float32)
    

    ## This function allows for tuple unpacking
    def __iter__(self):
        return iter((self.mean, self.cov))

class EKF():
    def __init__(self, dynamics_model, measurement_model):
        self.dyn_model = dynamics_model
        self.meas_model = measurement_model

    def predict(self, gauss_state, T):
        mu, S = gauss_state
        F = self.dyn_model.F(mu, T)

        mu_nxt = self.dyn_model.f(mu, T)
        Sigma_nxt = F@S@F.T + self.dyn_model.Q(mu, T)
    
        return GaussState(mu_nxt, Sigma_nxt)

    def update(self, gauss_state, y):
        mu, S = gauss_state
        y_hat = self.meas_model.h(mu)
        C = self.meas_model.H(mu)
        R = self.meas_model.R(mu)


        mu_nxt = mu+S@C.T@np.linalg.solve(C@S@C.T + R, y-y_hat)
        Sigma_nxt = S - S@C.T@np.linalg.solve(C@S@C.T + R, C@S)

        return GaussState(mu_nxt, Sigma_nxt)
