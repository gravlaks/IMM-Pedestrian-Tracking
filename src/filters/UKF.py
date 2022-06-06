from calendar import c
import numpy as np
from utils.Gauss import GaussState

import scipy.stats

class UKF():
    def __init__(self, dynamics_model, measurement_model, n=7):
        self.dyn_model = dynamics_model
        self.meas_model = measurement_model
        self.n = n

        ## Define weights:
        self.lambd = 2 
        self.w0 = self.lambd/(self.lambd+self.n)
        self.wi = 1/(2*(self.lambd+self.n))

    def get_sigma_points(self, mu, cov):
        x0 = mu.reshape((-1, 1))
        xs = [x0]
        try:
            cov_sqrt = np.linalg.cholesky(cov)
        except Exception as e:
            #print("non pos def")
            cov_sqrt = np.linalg.cholesky(cov+np.eye(cov.shape[0])*1e-30)

        for i in range(self.n):
            col = cov_sqrt[:, i].reshape((-1, 1))
            xs.append(
                (mu+np.sqrt(self.lambd+self.n)*col).reshape((-1, 1))
            )
            xs.append(
                (mu-np.sqrt(self.lambd+self.n)*col).reshape((-1, 1))
            )
        return xs
    def wrap_angle(self, x):
        while x>np.pi:
            x -= np.pi*2
        while x<=-np.pi:
            x += np.pi*2
        return x
    def inverse_UT(self, sigma_pts):
        mu_nxt = self.w0*sigma_pts[0].reshape((-1, 1)) + np.sum(np.array([self.wi*pt for pt in sigma_pts[1:]]), axis=0)
        cov_nxt = self.w0*np.outer(sigma_pts[0]-mu_nxt, sigma_pts[0]-mu_nxt)
        for pt in sigma_pts[1:]:
            cov_nxt += self.wi*np.outer(pt-mu_nxt, pt-mu_nxt)
        return mu_nxt, cov_nxt

    def S_XY(self, sigma_pts, sigma_pts_meas):
        mu_x = self.w0*sigma_pts[0].reshape((-1, 1)) + np.sum(np.array([self.wi*pt for pt in sigma_pts[1:]]), axis=0)
        mu_y = self.w0*sigma_pts_meas[0].reshape((-1, 1)) + np.sum(np.array([self.wi*pt for pt in sigma_pts_meas[1:]]), axis=0)

        #mu_x = self.w0*sigma_pts[0] + sum([self.wi*pt for pt in sigma_pts[1:]])
        
        #mu_y = self.w0*sigma_pts_meas[0] + sum([self.wi*pt for pt in sigma_pts_meas[1:]])
        mu_x, mu_y = mu_x.reshape((-1, 1)), mu_y.reshape((-1, 1))

        innov = sigma_pts_meas[0]-mu_y
        innov[1] = self.wrap_angle(innov[1])
        cov_xy = self.w0*np.outer(sigma_pts[0]-mu_x, innov)
        for pt_x, pt_y in zip(sigma_pts[1:], sigma_pts_meas[1:]):
            innov = pt_y-mu_y
            innov[1] = self.wrap_angle(innov[1])
            cov_xy += self.wi*np.outer(pt_x-mu_x, innov )
        return cov_xy

    def predict(self, gauss_state, u, T):



        mu, S = gauss_state
        sigma_points = self.get_sigma_points(mu, S)
        sigma_points_pred = [
            self.dyn_model.f(sigma_pt, u, T) for sigma_pt in sigma_points
        ]
        

        mu_nxt, S_nxt = self.inverse_UT(sigma_points_pred)
    
        return GaussState(mu_nxt, S_nxt+self.dyn_model.Q(x=None, u=u, T=T))

    def update(self, gauss_state, y):
        mu, S = gauss_state
        sigma_points = self.get_sigma_points(mu, S)
        sigma_points_meas = [
            self.meas_model.h(sigma_pt) for sigma_pt in sigma_points
        ]
        
        mu_y, S_Y = self.inverse_UT(sigma_points_meas)
        S_Y += self.meas_model.R(mu)


        S_xy = self.S_XY(sigma_points, sigma_points_meas)

        innov = y-mu_y
        innov[1] = self.wrap_angle(innov[1])

        K = S_xy@np.linalg.inv(S_Y)
        # K = S@np.linalg.solve((C@S@C.T+R), C).T

        mu_nxt = mu+K@innov
        C = self.meas_model.H(mu_nxt)
        KC  =  K@C
        I = np.eye(((KC).shape[0]))
        R = self.meas_model.R(mu)
        Sigma_nxt = (I - KC) @ S @ (I- KC).T + K@ R @ K.T
        #Sigma_nxt  = S -K@S_Y@K.T
        #Sigma_nxt = S - S_xy@np.linalg.solve(S_Y, S_xy.T)

        return GaussState(mu_nxt, Sigma_nxt)


    def loglikelihood(self, gauss_state, y):

        mu, S = gauss_state
        sigma_points = self.get_sigma_points(mu, S)
        sigma_points_meas = [
            self.meas_model.h(sigma_pt) for sigma_pt in sigma_points
        ]
        
        mu_y, S_Y = self.inverse_UT(sigma_points_meas)
        S_Y += self.meas_model.R(mu_y)
        
        innov = y-mu_y
        innov[1] = self.wrap_angle(innov[1])
       

        ll = scipy.stats.multivariate_normal.logpdf(innov.flatten(), cov=S_Y, allow_singular=True)
        if ll< -10:
            print(ll)
        return ll