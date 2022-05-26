from nbformat import read
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dynamics_models.CV_7dim import CV_7dim
from dynamics_models.CA_7dim import CA_7dim
from dynamics_models.CT_7dim import CT_7dim
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
from plot_statistics import plot_statistics
from utils.plotting import plot_trajectory

np.random.seed(13)

data = 'synthetic'
filter_model = 'CT' #options: CV, CT, CA
traj_num = 0
T = 0.1
N = 1000
sigma_q = 0.01
sigma_r = 0.1
sigma_th = 0.01
sigma_a = 0.05
sigma_w = 0.01

if data == 'synthetic':
    dt=0.1
    # X, zs = generate_data(N=N, dt=dt, mu0=np.zeros((7, 1)), cov0 = np.eye(7), process_noise=True, sensor_noise=True, run_model=filter_model)
    X, zs = (np.load('x.npy'), np.load('z.npy'))
    init_mean1 = (X[0] + np.random.randn(7, 1)).reshape(-1, 1)
    init_mean1[2:] = np.array([[0], [0], [0], [0], [0]])
if data == 'ped_dataset':
    dt = 1/30
    X, zs = get_data(traj_num, process_noise=False, sensor_noise=True)
    N, _ = X.shape
    init_mean1 = X[0,:]
    init_mean2 = X[0,:]

init_cov1 = np.eye((7))*1e-1

if filter_model == 'CV':
    filters = EKF(CV_7dim(sigma_q), RangeBearing(sigma_r, sigma_th, state_dim=7))
elif filter_model=='CA':
    filters = EKF(CA_7dim(sigma_q), RangeBearing(sigma_r, sigma_th, state_dim=7))
elif filter_model=='CT':
    # filters = EKF(CA_7dim(sigma_q), RangeBearing(sigma_z, state_dim=7))
    # filters = EKF(CV_7dim(sigma_q), RangeBearing(sigma_z, state_dim=7))
    filters = EKF(CT_7dim(sigma_a=sigma_a, sigma_w=sigma_w), RangeBearing(sigma_r, sigma_th, state_dim=7))

gauss0 = GaussState(init_mean1, init_cov1)

gaussStates     = [gauss0]
model_weights   = []
gauss = gauss0
for i in tqdm(range(1, N)):

    z = zs[i]

    pred       = filters.predict(gauss, u=None, T=dt)
    gauss      = filters.update(pred, z)
    # print(gauss.mean)
    gaussStates.append(gauss)
    
mus    = []
Sigmas = []

for gauss in gaussStates:
    mus.append(gauss.mean)
    Sigmas.append(gauss.cov)

mus    = np.array(mus)
Sigmas = np.array(Sigmas)

plot_trajectory(X, zs)

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

times = dt*np.ones(N)
times = np.cumsum(times)

plot_statistics(times, mus.squeeze(), Sigmas.squeeze(), X.squeeze(), zs.squeeze(), ['px', 'py', 'vx', 'vy', 'ax', 'ay', 'omega'], filter_model, meas_states=None)

plt.show()