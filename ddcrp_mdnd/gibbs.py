from ddcrp_mdnd.network_model import *
from ddcrp_mdnd.my_util import *
from ddcrp_mdnd.helper import *
import matplotlib.pyplot as plt
import numpy as np
def gibbs(X_train, alpha, gamma, tau, sample_params=True, maxit=1000):
    decay_fn = identity_decay()
    distance = linear_distance

    params = Params(alpha, gamma, tau, sample_params=sample_params, Ztrain=X_train)
    model = Model(X_train, distance, decay_fn, params, init=None)
    burnin = 500
    samples = np.zeros((maxit-burnin,(len(X_train))), dtype=float)
    for iter in range(maxit):
        for edge in range(model.num_edges):
            print(model.cluster_of_edge)
            if iter >= burnin:
                samples[iter-burnin,:] = model.cluster_of_edge
            linked_edges = model.remove_link(edge, params)
            model.sample_link_collapsed(edge, params, linked_edges)
            model.sample_beta()

            # if params.sample_params is True:
            #     params.sample_alpha

    return model, params, np.array(samples)


if __name__ == '__main__':
    from data.get_data import get_data_simple, display_adjacency
    data, nodes, _ = get_data_simple('../data/sbm')
    print(data.shape)
    _,_,samples = gibbs(data, alpha=5, gamma=1, tau=1, sample_params=False)
    # for i in range(len(data)):
    #     plt.subplot(3,4,i+1)
    #     plt.hist(samples[:,i])
    # plt.show()
