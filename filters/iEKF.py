import numpy as np
from utils.Gauss import GaussState

class iEKF():
    def __init__(self, dynamics_model, measurement_model):
        self.dyn_model = dynamics_model
        self.meas_model = measurement_model

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
        eps = 1e-1
        iter_Max = 1000
        for _ in range(iter_Max):
            mu_nxt = (mu+
                    S@C.T@np.linalg.solve(C@S@C.T + R, y-y_hat)
                    + S@C.T@np.linalg.solve(C@S@C.T + R, C@(mu_i-mu))
            )

            if np.linalg.norm(mu_nxt-mu_i)<eps:
                break
            y_hat = self.meas_model.h(mu)
            C = self.meas_model.H(mu)
            mu_i = mu_nxt
        Sigma_nxt = S - S@C.T@np.linalg.solve(C@S@C.T + R, C@S)

        return GaussState(mu_nxt, Sigma_nxt)
