from ddcrp_mdnd.my_util import *
from ddcrp_mdnd.helper import *
from ddcrp_mdnd.network_model import *
import numpy as np

decay_fn = identity_decay()
distance = linear_distance
alpha = 5
gamma = 1
tau = 1
X_train = np.array([[0, 0, 1, 2, 3, 2], [4, 3, 0, 0, 1, 2]]).reshape((6,2))
init = np.array([2,0,1,3,5,3])
params = Params(alpha, gamma, tau, sample_params=False, Ztrain=X_train)
model = Model(X_train, distance, decay_fn, params, init=init)

model.sample_beta()
print(model.beta)
# for edge in range(model.num_edges):
#     linked_edges = model.remove_link(edge, params)
#     model.sample_link_collapsed(edge, params, linked_edges)
#     print(model.cluster_of_edge.tolist())
#     model.sample_beta()
