from nbformat import read
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dynamics_models.CV_inc import CV
from dynamics_models.CA import CA
from measurement_models.range_only import RangeOnly
from measurement_models.range_bearing import RangeBearing
from filters.EKF import EKF
from filters.UKF import UKF
from filters.iEKF import iEKF
from utils.Gauss import GaussState
from read_data import read_data
from imm.Gaussian_Mixture import GaussianMixture
from imm.IMM import IMM
T = 0.1
N = 500
sigma_q = 0.1
sigma_z = 0.1

dyn_mod = CA(sigma_q, n=2)
filters = [
    EKF(CA(sigma_q, n=2), RangeBearing(sigma_z, state_dim=6)),
    EKF(CV(sigma_q, n=2), RangeBearing(sigma_z, state_dim=6))
]
init_weights = np.ones((2, 1))/2.

init_mean1 = np.zeros((6, 1))
init_mean2 = np.zeros((6, 1))
init_cov1 = np.eye((6))*1.001
init_cov2 = np.eye((6))

init_states = [
    GaussState(init_mean1, init_cov1),
        GaussState(init_mean2, init_cov2)]

immstate = GaussianMixture(
    init_weights, init_states
)

##High probability that you stay in state
PI = np.array([[0.95, 0.05],
             [0.05, 0.95]])

imm = IMM(filters, PI)

GT, Z = read_data()

GT, Z = GT[:N], Z[:N]
Ts = GT[:, 0]
X = GT[:, 1:3]
previous_time = 0
gaussStates = []

for i in tqdm(range(1, N-1)):
    dt = Ts[i]-previous_time

    z = np.array([
        np.linalg.norm(X[i], 2)+np.random.randn(1)*0.1, 
        np.arctan(X[i, 1]/X[i, 0])+np.random.randn(1)*0.01]
    ).reshape((-1, 1))
    #+np.random.randn(1)*10

    immstate = imm.predict(immstate,u=None, T=dt)
    immstate = imm.update(immstate, z)
    gaussStates.append(imm.get_estimate(immstate))

    previous_time = Ts[i]
    
mus = []
for gauss in gaussStates:
    mus.append(gauss.mean)
mus = np.array(mus)
plt.plot(X[:, 0], X[:, 1], label="GT")
plt.plot(mus[:, 0], mus[:, 1], label="Estimate")
plt.legend()
plt.show()