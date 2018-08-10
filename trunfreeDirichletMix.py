from matplotlib import colors
import numpy as np
from numpy.random import choice, dirichlet, choice, multinomial, permutation, gamma
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from scipy.special import psi
import copy

# mixture of exchangable edges with INfinite clusters using truncation-free variational inference

def get_data(fname):
    """get data edge list from file. should return an (n,2) array of edges"""
    f = open(fname, 'r')
    res = np.array([list(map(int, line.strip().split(' '))) for line in f], dtype=int)
    nodes = list(set(res.flatten()))
    sz = int(res.flatten().max() + 1)
    adj = np.zeros((sz, sz), dtype=int)

    for e in res:
        adj[e[0], e[1]] += 1

    return res, nodes, adj


def trunfreeInfiniteClusterDirichletMix(n_clusters, edges, edges_test):
    """replace HDP with mixture of Dirichlet-Categorical"""
    n_edges = edges.shape[0]
    n_nodes = len(nodes)

    # store variable that will be sampled
    state = {
        'n_clusters': n_clusters, # can change!
        # 'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
        # the cluster assignment of each edge
        # 'sizes': Counter({i: 0 for i in range(n_clusters)}),  # size of each cluster
        'assignments': np.zeros(n_edges)
    }

    # store hyperparameters (unchanged)
    fixed = {
        'alpha': 1.0, # Beta prior for stick breaking ~ Beta(1, \alpha)
        'gamma': np.array([0.1]*len(nodes), dtype=float), # Dirichlet prior for node cluster weight  \theta ~ Dir(\gamma)
        'edges': edges,
        'n_edges': n_edges,
        'n_nodes': len(nodes),
        'node_ids': None,
    }

    vi = {
        'h_a': np.zeros((n_nodes, n_clusters), dtype=float), # vi for dirichlet prior of A_k p(A_k|h_a) = Dir(h_a)
        'h_b': np.zeros((n_nodes, n_clusters), dtype=float), # vi for dirichlet prior of B_k
        'h_sticks': np.zeros((2, n_clusters), dtype=float), # the stick parameters a_k and b_k, can be expanded as new clusters are needed (truncate???)
        'lambda_sum': None # for optimal reordering, is the cluster size in [h_a,h_b]
    }

    # train parameters
    train = {
        'kappa': 0.7, # robbins-monro
        'tau': 0, # cooler
        'n_iter': 8000,
        'n_validate': 200,
        'batch_size': 100,
        'n_sample': 100,
        'eps': 0.001 # new cluster threshold
    }

    # def vi_c(ind):
    #     """
    #     update c_hat given an edge
    #     :return:
    #     """
    #
    #     u = fixed['edges'][ind,0]
    #     v = fixed['edges'][ind,1]
    #     logp = []
    #     for k in state['cluster_ids']:
    #         psi_alpha = psi(vi['h_alpha'][k])
    #         psi_a_k_u = psi(vi['h_a'][k][u])
    #         psi_a_k_un = psi(sum(vi['h_a'][k]))
    #         psi_a_k_v = psi(vi['h_b'][k][v])
    #         psi_a_k_vn = psi(sum(vi['h_b'][k]))
    #         # print(psi_alpha, psi_a_k_u, psi_a_k_un, psi_a_k_v, psi_a_k_vn)
    #         logpk = psi_alpha + psi_a_k_u - psi_a_k_un + psi_a_k_v - psi_a_k_vn
    #         logp += [logpk]
    #
    #     #underflow
    #     logp = logp - max(logp)
    #     p = np.exp(logp)
    #     return p / sum(p)

    # def get_local_estimate():
    #     p_h_c = []
    #     for ind,e in enumerate(edges):
    #         p_ind = vi_c(ind)
    #         p_h_c.append(p_ind)
    #         assert(abs(sum(p_ind) - 1) < 0.00001)
    #     vi['h_c'] = p_h_c
    #     return vi['h_c']

    def locally_collapsed_c_sample(ind):
        u = fixed['edges'][ind,0]
        v = fixed['edges'][ind,1]
        p = []
        beta_v_prod = 1 # \prod v_l / (u_l + v_l)

        # for k old
        for k in range(state['n_clusters']):
            p_u_k = (vi['h_a'][u,k]) / sum(vi['h_a'][:,k])
            p_v_k = (vi['h_b'][v,k]) / sum(vi['h_b'][:,k])
            stick_ratio_k = vi['h_sticks'][0,k] / (vi['h_sticks'][0,k] + vi['h_sticks'][1,k]) # u_k / (u_k + v_k)

            # marginal likelihood of DP
            p_k = p_u_k * p_v_k * stick_ratio_k * beta_v_prod
            beta_v_prod *= vi['h_sticks'][1,k] / (vi['h_sticks'][0,k] + vi['h_sticks'][1,k])

            p += [p_k]

        # for k new, use prior distributions
        p_u_k_new = fixed['gamma'][u] / sum(fixed['gamma'])
        p_v_k_new = fixed['gamma'][v] / sum(fixed['gamma'])
        stick_ratio_k_new = fixed['alpha'] / (1 + fixed['alpha']) * beta_v_prod
        p_k_new = p_u_k_new * p_v_k_new * stick_ratio_k_new

        p += [p_k_new]
        return p

    def get_map():
        """
        get point estimate for all cluster indicator
        :return:
        """
        map = []
        for ind in range(n_edges):

            u = fixed['edges'][ind,0]
            v = fixed['edges'][ind,1]
            p = []
            beta_v_prod = 1 # \prod v_l / (u_l + v_l)

            for k in range(state['n_clusters']):
                p_u_k = (vi['h_a'][u,k]) / sum(vi['h_a'][:,k])
                p_v_k = (vi['h_b'][v,k]) / sum(vi['h_b'][:,k])
                stick_ratio_k = vi['h_sticks'][0,k] / (vi['h_sticks'][0,k] + vi['h_sticks'][1,k]) # u_k / (u_k + v_k)

                # marginal likelihood of DP
                p_k = p_u_k * p_v_k * stick_ratio_k * beta_v_prod
                beta_v_prod *= vi['h_sticks'][1,k] / (vi['h_sticks'][0,k] + vi['h_sticks'][1,k])

                p += [p_k]

            map += [np.argmax(p)]

        return map

    def init_vi():
        """
        initialize vi parameters
        :return:
        """

        vi['h_a'] = gamma(1.0, 1.0, (n_nodes, state['n_clusters'])) * 100 / (n_nodes * state['n_clusters'])
        # vi['h_a'] = np.subtract(vi['h_a'], fixed['gamma'].reshape(n_nodes, 1))
        vi['h_b'] = gamma(1.0, 1.0, (n_nodes, state['n_clusters'])) * 100 / (n_nodes * state['n_clusters'])
        # vi['h_b'] = np.subtract(vi['h_b'], fixed['gamma'].reshape(n_nodes, 1))
        # vi['h_a'] = np.tile(fixed['gamma'].reshape((n_nodes,1)), (1,n_clusters)) + np.random.rand(n_nodes, n_clusters)
        # vi['h_b'] = np.tile(fixed['gamma'].reshape((n_nodes,1)), (1,n_clusters)) + np.random.rand(n_nodes, n_clusters)
        vi['h_sticks'][0] = 1
        vi['h_sticks'][1] = np.arange(state['n_clusters'],0,-1)
        # vi['h_sticks'] = np.tile(np.array([1,fixed['alpha']]).reshape((2,1)), (1,n_clusters)) + np.random.rand(2, n_clusters)


    def get_batch(batch_size):
        i = 0
        cands = permutation(n_edges)
        while i < n_edges:
            res = cands[i: i+batch_size]
            yield res, len(res)
            i += batch_size

    def add_new_cluster():
        """
        add new cluster
        :return:
        """

        #vi params
        state['n_clusters'] += 1
        vi['h_a'] = np.hstack((vi['h_a'], fixed['gamma'][:,np.newaxis]))
        vi['h_b'] = np.hstack((vi['h_b'], fixed['gamma'][:,np.newaxis]))
        vi['h_sticks'] = np.hstack((vi['h_sticks'], [[1],[fixed['alpha']]]))

        assert(vi['h_a'].shape == (fixed['n_nodes'], state['n_clusters']))
        assert(vi['h_sticks'].shape == (2, state['n_clusters']))

    def optimal_reordering():
        """
        reorder clusters
        :return:
        """
        vi['lambda_sum'] = vi['h_a'].sum(axis=0) + vi['h_b'].sum(axis=0)
        idx = [i for i in reversed(np.argsort(vi['lambda_sum']))]
        vi['lambda_sum'] = vi['lambda_sum'][idx]
        vi['h_a'] = vi['h_a'][:,idx]
        vi['h_b'] = vi['h_b'][:,idx]
        # vi['h_sticks'] = vi['h_sticks'][:,idx]

    def onehot(size, pos):
        res = np.zeros(size, dtype=int)
        res[pos] = 1
        return res

    def heldout_loglikelihood():
        """"""
        pass

    def vi_train_stochastic(batch_size=100, locally_collapsed=False):
        """
        update lambda, optimal ordering, then update stick
        :param batch_size:
        :param locally_collapsed:
        :return:
        """
        print('stochastic variational inference')
        init_vi()

        for iter in range(train['n_iter']):

            # update weight
            rho = (iter+1+train['tau'])**(-train['kappa'])

            for ind in range(n_edges):

                # locally collapsed vi update
                p_ind = locally_collapsed_c_sample(ind)
                # 1 sample for approximating expectation
                # emp = multinomial(1, p_ind)

                # c = choice(len(p_ind), p=p_ind)
                # state['assignments'][ind] = c
                # emp = onehot(len(p_ind), c)
                #
                # if emp[-1] == 1 and p_ind:

                # new cluster!
                thres = p_ind[-1] / sum(p_ind)
                if thres > train['eps']:
                    add_new_cluster()
                    emp = p_ind / sum(p_ind)
                else:
                    emp = p_ind[:-1] / sum(p_ind[:-1])

                for k in range(state['n_clusters']):
                    u = fixed['edges'][ind,0]
                    v = fixed['edges'][ind,1]

                    vi['h_a'][u,k] = (1-rho) * vi['h_a'][u,k] + rho * (fixed['gamma'][u] + n_edges * emp[k])
                    vi['h_b'][v,k] = (1-rho) * vi['h_b'][v,k] + rho * (fixed['gamma'][v] + n_edges * emp[k])

                # optimal_reordering()

                for k in range(state['n_clusters']):
                    vi['h_sticks'][0][k] = (1-rho) * vi['h_sticks'][0][k] + rho * (1 + n_edges * emp[k])
                    vi['h_sticks'][1][k] = (1-rho) * vi['h_sticks'][1][k] + rho * (fixed['alpha'] + n_edges * emp[k+1:].sum())

            print('train | iter {}'.format(iter))
            cluster = get_map()
            # cluster = state['assignments']
            cmap = colors.ListedColormap(['white', 'red', 'green', 'blue','yellow','purple','orange','brown','black'])
            bounds = [0,1,2,3,4,5,6,7,8,9]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            count = Counter(cluster).most_common()
            count = [c[0] for c in count]
            temp = [1,2,3,4,5,6,7,8,9]
            color = {count[x]: temp[x] if x < 8 else temp[8] for x in range(len(count))}
            for e,c in zip(edges, cluster):
                adj[e[0], e[1]] = color[c]
            plt.imshow(adj,cmap=cmap,norm=norm)
            plt.pause(1)

        plt.show()

    vi_train_stochastic(1, locally_collapsed=True)

    return state,vi


if __name__ == '__main__':
    edges, nodes, adj = get_data('sbm')
    n_edges = len(edges)
    plt.imshow(adj)

    state,vi = trunfreeInfiniteClusterDirichletMix(1, np.array(edges), np.array(nodes))

    # from main_test.run import get_data
    # links_train,links_test,clusters_train,clusters_test,nodes = get_data('main_test/toy_test')
    # state,vi = sequentialDN(np.array(links_train), np.array(nodes))
    # print(vi['h_c'])
    # print(vi['h_alpha'])
