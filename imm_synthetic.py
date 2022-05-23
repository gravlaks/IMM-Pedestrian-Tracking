from nbformat import read
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dynamics_models.CV_7dim import CV_7dim
from dynamics_models.CA_7dim import CA_7dim
from dynamics_models.CT_7dim import CT_7dim
from measurement_models.range_only import RangeOnly
from measurement_models.range_bearing import RangeBearing
# from measurement_models.range_bearing import RangeOnly
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

data = 'synthetic'
traj_num = 154
T = 0.1
N = 500
sigma_q = 0.1
sigma_z = 0.1
sigma_a = 0.1
sigma_w = 0.01

np.random.seed(seed=1)

sensor_model = RangeBearing(sigma_z, state_dim=7)

if data == 'synthetic':
    dt=0.1
    # X, zs = generate_data(N=N, dt=dt, mu0=np.zeros((7, 1)), cov0 = np.eye(7), process_noise=False, sensor_noise=False)
    X, zs, switches = (np.load('x.npy'), np.load('z.npy'), np.load('switches.npy'))
    print(switches*dt)
    init_mean1 = np.zeros((7, 1))
    init_mean2 = np.zeros((7, 1))
    N = len(X)
if data == 'ped_dataset':
    dt = 1/30
    X, zs = get_data(traj_num, sensor_model, process_noise=False, sensor_noise=True)
    N, _ = X.shape
    init_mean1 = X[0,:]
    init_mean2 = X[0,:]

init_cov1 = np.eye((7))*1.001
init_cov2 = np.eye((7))

# filters = [
#     UKF(CV(sigma_q, n=2), sensor_model),
#     UKF(CA(sigma_q, n=2), sensor_model),
# ]
filters = [
    EKF(CV_7dim(sigma_q), sensor_model),
    EKF(CT_7dim(sigma_a=sigma_a, sigma_w=sigma_w), sensor_model),
]

n_filt = len(filters)
init_weights = np.ones((n_filt, 1))/n_filt

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

gauss0, _       = imm.get_estimate(immstate)
gaussStates     = [gauss0]
model_weights   = []
for i in tqdm(range(1, N)):

    z = zs[i]

    immstate       = imm.predict(immstate,u=None, T=dt)
    gauss, weights = imm.get_estimate(immstate)
    # print(gauss.mean)
    immstate       = imm.update(immstate, z)
    gauss, weights = imm.get_estimate(immstate)
    # print(gauss.mean)
    gaussStates.append(gauss)
    model_weights.append(weights.flatten())

    
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

plt.figure()

for i in range(model_weights.shape[1]):

    plt.plot(model_weights[:,i],
         label= filters[i].dyn_model.__class__.__name__
    )
plt.legend()
plt.title('Model Weights')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.grid(True)

times = dt*np.ones(N)
times = np.cumsum(times)

plot_statistics(times, mus.squeeze(), Sigmas.squeeze(), X.squeeze(), zs.squeeze(), ['px', 'py', 'vx', 'vy', 'ax', 'ay', 'w'], 'imm', meas_states=None)

plt.show()