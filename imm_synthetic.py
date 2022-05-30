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
from get_ped_data import get_data
from plot_statistics import plot_statistics
from plot_errors import plot_errors

data = 'synthetic'
traj_num = 154
T = 0.1
N = 500

#cv
# sigma_q = 0.2

# #measurement noise
# sigma_r = 0.5
# sigma_th = 0.05

# sigma_a = 0.25
# sigma_w = 0.01

sigma_q = 0.001
sigma_r = 0.1
sigma_th = 0.001
sigma_a = 0.3
sigma_w = 0.02

np.random.seed(seed=12)

sensor_model = RangeBearing(sigma_r, sigma_th, state_dim=7)
filters = [
    UKF(CV_7dim(sigma_q), sensor_model),
    UKF(CT_7dim(sigma_a, sigma_w=sigma_w), sensor_model),
    UKF(CA_7dim(sigma=sigma_a), sensor_model),

    EKF(CV_7dim(sigma_q), sensor_model),
    EKF(CT_7dim(sigma_a, sigma_w=sigma_w), sensor_model),
    EKF(CA_7dim(sigma=sigma_a), sensor_model)
]

individual_filters = [
    UKF(CV_7dim(sigma_q), sensor_model),
    UKF(CT_7dim(sigma_a, sigma_w=sigma_w), sensor_model),
    UKF(CA_7dim(sigma=sigma_a), sensor_model),

    EKF(CV_7dim(sigma_q), sensor_model),
    EKF(CT_7dim(sigma_a, sigma_w=sigma_w), sensor_model),
    EKF(CA_7dim(sigma=sigma_a), sensor_model)
]

filter_names = [filt.__class__.__name__ + filt.dyn_model.__class__.__name__ for filt in filters]

n_filt = len(filters)
if data == 'synthetic':
    init_means = []
    dt=0.1
    # X, zs = generate_data(N=N, dt=dt, mu0=np.zeros((7, 1)), cov0 = np.eye(7), process_noise=False, sensor_noise=False)
    X, zs, switches = (np.load('x.npy'), np.load('z.npy'), np.load('switches.npy'))
    print(switches*dt)
    for i in range(n_filt):
    # init_mean1 = np.zeros((7, 1))
    # init_mean2 = np.zeros((7, 1))
        init_mean1 = (X[0]).reshape(-1, 1)
        init_mean1[:6] += np.random.randn(6, 1)*0.1
        init_mean1[6] +=np.random.randn()*1e-2
        init_means.append(init_mean1)
    

    N = len(X)
if data == 'ped_dataset':
    dt = 1/30
    X, zs = get_data(traj_num, sensor_model, process_noise=False, sensor_noise=True)
    N, _ = X.shape
    init_mean1 = X[0,:]
    init_mean2 = X[0,:]

init_covs = []
for i in range(n_filt):
    init_cov1 = np.eye((7))*0.1
    init_cov1[6, 6] = 1e-2
    init_covs.append(init_cov1)
init_mean1, init_mean2 = init_means[0], init_means[1]
init_cov1, init_cov2 = init_covs[0], init_covs[1]


# filters = [
#     UKF(CV(sigma_q, n=2), sensor_model),
#     UKF(CA(sigma_q, n=2), sensor_model),
# ]

init_weights = np.ones((n_filt, 1))/n_filt

init_states = [GaussState(init_mean, init_cov) for init_mean, init_cov in zip(init_means, init_covs)]

immstate = GaussianMixture(
    init_weights, init_states
)

##High probability that you stay in state
alpha = 0.95
def generate_pi():
    Pi = np.eye(n_filt)*alpha
    for i in range(n_filt):
        for j in range(n_filt):
            if i != j:
                Pi[i, j] = (1-alpha)/(n_filt-1)
    return Pi
PI = generate_pi()
# PI = np.array([[alpha, (1-alpha)],
# [alpha, (1-alpha)]
# ])
imm = IMM(filters, PI)

individual_gauss_states = {}
for flt in individual_filters:
	individual_gauss_states[flt] = [GaussState(init_mean1, init_cov1)]

gauss0, _       = imm.get_estimate(immstate)
gaussStates     = [gauss0]
model_weights   = []
for i in tqdm(range(1, N)):

    z = zs[i]

    immstate       = imm.predict(immstate,u=None, T=dt)
    gauss, weights = imm.get_estimate(immstate)
    immstate       = imm.update(immstate, z)
    gauss, weights = imm.get_estimate(immstate)
    gaussStates.append(gauss)
    model_weights.append(weights.flatten())

    for flt in individual_filters:
    	pred = flt.predict(individual_gauss_states[flt][-1], u=None, T=dt)
    	upd  = flt.update(pred, z)
    	individual_gauss_states[flt].append(upd)

    
mus    = []
Sigmas = []

model_weights = np.array(model_weights)
print(model_weights.shape)

for gauss in gaussStates:
    mus.append(gauss.mean)
    Sigmas.append(gauss.cov)

mus    = np.array(mus)
Sigmas = np.array(Sigmas)

mus_ind = {}
Sigmas_ind = {}

for flt in individual_filters:
	mus_ind[flt] = []
	Sigmas_ind[flt] = []
	for gauss in individual_gauss_states[flt]:
		mus_ind[flt].append(gauss.mean)
		Sigmas_ind[flt].append(gauss.cov)

	mus_ind[flt] = np.array(mus_ind[flt]).squeeze()
	Sigmas_ind[flt] = np.array(Sigmas_ind[flt]).squeeze()

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
         label= filter_names[i]
    )
plt.legend()
plt.title('Model Weights')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.grid(True)

times = dt*np.ones(N)
times = np.cumsum(times)

plot_statistics(times, mus.squeeze(), Sigmas.squeeze(), X.squeeze(), zs.squeeze(), ['Px', 'Py', 'Vx', 'Vy', 'Ax', 'Ay', 'Turn Rate'], 'IMM', meas_states=None)

mus_ind['imm'] = mus.squeeze()
Sigmas_ind['imm'] = Sigmas.squeeze()

# for idx, flt in enumerate(individual_filters):
# 	savestr=filter_names[idx]
# 	plot_statistics(times, mus_ind[flt], Sigmas_ind[flt], X.squeeze(), zs.squeeze(), ['Px', 'Py', 'Vx', 'Vy', 'Ax', 'Ay', 'Turn Rate'], savestr=savestr, meas_states=None)

filter_names.append('IMM')
plot_errors(times, mus_ind, Sigmas_ind, X.squeeze(), zs.squeeze(), filter_names, 'imm_errors', switches=switches*dt)

plt.show()