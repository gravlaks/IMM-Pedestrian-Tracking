from nbformat import read
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dynamics_models.CA_7dim import CA_7dim
from dynamics_models.CT_7dim import CT_7dim
from dynamics_models.CT_7dim_alt import CT_7dim_alt

from dynamics_models.CV_inc import CV
from dynamics_models.CA import CA
from dynamics_models.CV_7dim import CV_7dim
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
N = 1000
sigma_q = 0.1
sigma_z = 0.1
sigma_w = 0.01
n = 7

filters = [
    EKF(CT_7dim(sigma_q, sigma_w), RangeBearing(sigma_z*1, sigma_th=0.01, state_dim=n)),
    EKF(CA_7dim(sigma_q), RangeBearing(sigma_z*1, sigma_th=0.01,state_dim=n)),
    EKF(CV_7dim(sigma_q), RangeBearing(sigma_z*1, sigma_th=0.01,state_dim=n)),
]

init_weights = np.ones((len(filters), 1))/len(filters)
init_mean1 = np.zeros((n, 1))
init_mean2 = np.zeros((n, 1))
init_cov1 = np.eye((n))*1.001
init_cov2 = np.eye((n))

init_states = [
    GaussState(init_mean1, init_cov1),
    GaussState(init_mean1, init_cov1),
   # GaussState(init_mean1, init_cov1),
   # GaussState(init_mean1, init_cov1),
    GaussState(init_mean2, init_cov2), 
        ]

immstate = GaussianMixture(
    init_weights, init_states
)

##High probability that you stay in state
PI = np.array([[0.95, 0.02, 0.03],
             [0.02, 0.95, 0.03, ],
             #[0.02, 0.01, 0.95, 0.01, 0.01],
            # [0.02, 0.01, 0.01, 0.95, 0.01],
             [0.02, 0.03, 0.95]])

imm = IMM(filters, PI)

GT, Z = read_data()

GT, Z = GT[:N], Z[:N]
Ts = GT[:, 0]
X = GT[:, 1:3]
previous_time = 0
gaussStates = []
model_weights = []
for i in tqdm(range(1, N-1)):
    dt = Ts[i]-previous_time

    z = np.array([
        np.linalg.norm(X[i], 2)+np.random.randn(1)*0.1, 
        np.arctan(X[i, 1]/X[i, 0])+np.random.randn(1)*0.001]
    ).reshape((-1, 1))
    #+np.random.randn(1)*10

    immstate = imm.predict(immstate,u=None, T=dt)
    immstate = imm.update(immstate, z)
    gauss, weights = imm.get_estimate(immstate)
    gaussStates.append(gauss)
    model_weights.append(weights.flatten())

    previous_time = Ts[i]
    
mus = []
model_weights = np.array(model_weights)
print(model_weights.shape)
for gauss in gaussStates:
    mus.append(gauss.mean)
mus = np.array(mus)
plt.plot(X[:, 0], X[:, 1], label="GT")
plt.plot(mus[:, 0], mus[:, 1], label="Estimate")
print(N)
for i in range(N):
    T = f"{(i*dt):.0f}"
    if i % (N//10) == 0:
        plt.annotate(T, (mus[i, 0], mus[i, 1]))

plt.legend()
plt.show()

for i in range(model_weights.shape[1]):

    plt.plot(model_weights[:,i],
         label= filters[i].dyn_model.__class__.__name__
    )
plt.legend()
plt.show()