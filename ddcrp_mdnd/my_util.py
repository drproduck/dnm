import numpy as np
from scipy.special import gammaln
from collections import Counter
import networkx as nx

def dirichlet_likelihood(Xp, hyper):
    if len(Xp.shape) == 2:
        X = sum(Xp)
    else:
        X = Xp
    idx = np.where(X > 0)
    lh = gammaln(len(X) * hyper) \
         - len(idx[0]) * gammaln(hyper) \
         + sum(gammaln(X[idx] + hyper)) \
         - gammaln(sum(X[idx]) + len(X) * hyper)

    return lh


def connected_component(i, graph):
    """
    get the customers linked to customer i
    since there are n customers and n links, the number of operations is exactly n
    can we do better?
    :param i:
    :param link:
    :return list of customers that are linked to i
    """

    return list(nx.node_connected_component(graph, i))


# def hdp_llhood(obs, base, params):
#     count = Counter()
#     lhood = 0
#     for i in range(len(obs)):
#         lhood += np.log(count[obs[i]] + params.tau * base[obs[i]]) - np.log(i + params.tau)
#         count[obs[i]] += 1
#     return lhood


class Params(object):
    def __init__(self, alpha_params, gamma_params, tau_params, sample_params, Ztrain):
        self.num_users = np.max(Ztrain) + 1
        self.num_links = len(Ztrain)
        self.riters = 10
        self.test_iters = 1000
        self.test_sample_every = 100
        if sample_params is True:
            self.alpha_params = alpha_params
            self.gamma_params = gamma_params
            self.tau_params = tau_params
            self.alpha = alpha_params[0] / alpha_params[1]
            self.gamma = gamma_params[0] / gamma_params[1]
            self.tau = tau_params[0] / tau_params[1]
            self.sample_params = sample_params
        else:
            self.alpha = alpha_params
            self.gamma = gamma_params
            self.tau = tau_params
            self.sample_params = sample_params

    @staticmethod
    def sample_hdp_hyper(alpha, num_groups, num_per_group, num_tables, a, b, niters=1):
        for iter in range(niters):
            w = np.zeros(num_groups)
            s = np.zeros(num_groups)

            for j in range(num_groups):
                w[j] = np.random.beta(alpha + 1, num_per_group[j])
                pie = num_per_group[j] / (num_per_group[j] + alpha)
                r = np.random.rand()
                if r < pie:
                    s[j] = 1
            gam_a = a + num_tables - np.sum(s)
            gam_b = b - np.sum(np.log(w))
            # pdb.set_trace()
            alpha = np.random.gamma(gam_a, 1) / gam_b
        return alpha

    @staticmethod
    def sample_dirichlet_hyper(alpha, N, K, a, b, niters=1):
        for n in range(niters):
            eta = np.random.beta(alpha + 1, N)
            pie = (a + K - 1) / (a + K - 1 + N * (b - np.log(eta)))
            r = np.random.rand()
            if r < pie:
                alpha = np.random.gamma(a + K, 1 / (b - np.log(eta)))
            else:
                alpha = np.random.gamma(a + K - 1, 1 / (b - np.log(eta)))
        return alpha

    def sample_alpha(self, num_clusters):
        if self.sample_params is True:
            self.alpha = self.sample_dirichlet_hyper(self.alpha, self.num_links, num_clusters, self.alpha_params[0],
                                                     self.alpha_params[1], niters=10)

    def sample_tau(self, num_clusters, num_per_cluster, num_tables):
        if self.sample_params is True:
            num_groups = 2 * num_clusters
            num_per_group = np.hstack((num_per_cluster, num_per_cluster))
            self.tau = self.sample_hdp_hyper(self.tau, num_groups, num_per_group, num_tables, self.tau_params[0],
                                             self.tau_params[1], niters=10)

    def sample_gamma(self, num_tables):
        if self.sample_params is True:
            self.gamma = self.sample_dirichlet_hyper(self.gamma, num_tables, self.num_users, self.gamma_params[0],
                                                     self.gamma_params[1], niters=10)
