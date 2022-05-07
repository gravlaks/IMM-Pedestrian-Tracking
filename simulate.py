from nbformat import read
import numpy as np
import matplotlib.pyplot as plt

from dynamics_models.CV import CV
from dynamics_models.CA import CA
from measurement_models.range_only import RangeOnly
from measurement_models.range_bearing import RangeBearing
from filters.EKF import EKF, GaussState
from read_data import read_data


T = 0.1
N = 500
sigma_q = 10
sigma_z = 10

dyn_mod = CV(sigma_q, n=2)
meas_mod = RangeBearing(sigma_z, m=1, n=2)

mu0 = np.ones((4,1))
cov0 = np.eye(4)*10

gauss0 = GaussState(mu0, cov0)
ekf_filter = EKF(dyn_mod, meas_mod)

GT, Z = read_data()

GT, Z = GT[:N], Z[:N]
Ts = GT[:, 0]
X = GT[:, 1:3]
print(Z)

gaussStates = [gauss0]
previous_time = Ts[0]
gauss = gauss0
for i in range(1, N-1):
    dt = Ts[i]-previous_time

    z = np.array([
        np.linalg.norm(X[i], 2)+np.random.randn(1)*0.1, 
        np.arctan(X[i, 1]/X[i, 0])+np.random.randn(1)*0.001]
    ).reshape((-1, 1))
    print(z)
    #+np.random.randn(1)*10

    print(z)
    pred = ekf_filter.predict(gauss, dt)
    gauss = ekf_filter.update(pred, z)
    gaussStates.append(gauss)

    previous_time = Ts[i]
    
mus = []
for gauss in gaussStates:
    mus.append(gauss.mean)
mus = np.array(mus)
plt.plot(X[:, 0], X[:, 1], label="GT")
plt.plot(mus[:, 0], mus[:, 1], label="Estimate")
plt.legend()
plt.show()