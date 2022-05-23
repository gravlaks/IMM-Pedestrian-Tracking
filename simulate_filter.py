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
from generate_synthetic import generate_data
from generate_synthetic import get_data

data = 'ped_dataset'
traj_num = 154
T = 0.1
N = 500
sigma_q = 0.1
sigma_z = 0.1

np.random.seed(seed=1)

sensor_model = RangeBearing(sigma_z, state_dim=6)

if data == 'synthetic':
    dt=0.1
    X, zs = generate_data(N=N, dt=dt, mu0=np.zeros((6, 1)), cov0 = np.eye(6), process_noise=True, sensor_noise=False)
    init_mean1 = np.zeros((6, 1))
    init_mean2 = np.zeros((6, 1))
if data == 'ped_dataset':
    dt = 1/30
    X, zs = get_data(traj_num, sensor_model, process_noise=False, sensor_noise=True)
    N, _ = X.shape
    init_mean = X[0,:]

init_cov = np.eye((6))*1.001

filt = UKF(CA(sigma_q, n=2), RangeBearing(sigma_z, state_dim=6))

init_states = GaussState(init_mean, init_cov)

gaussStates     = [init_states]
model_weights   = []
for i in tqdm(range(1, N)):

    z = zs[i]

    gauss       = filt.predict(gaussStates[i-1],u=None, T=dt)
    # print(gauss.mean)
    gauss       = filt.update(gauss, z)
    # print(gauss.mean)
    gaussStates.append(gauss)
    
mus    = []
Sigmas = []

model_weights = np.array(model_weights)
print(model_weights.shape)

for gauss in gaussStates:
    mus.append(gauss.mean)
    Sigmas.append(gauss.cov)

mus    = np.array(mus)
Sigmas = np.array(Sigmas)

plt.plot(X[:, 0], X[:, 1], label="GT")
plt.plot(mus[:, 0], mus[:, 1], label="Estimate")
print(N)
for i in range(N):
    T = f"{(i*dt):.0f}"
    if i % (N//10) == 0:
        plt.annotate(T, (mus[i, 0], mus[i, 1]))

plt.legend()
plt.title('Filter vs. True Traj')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')

plt.show()