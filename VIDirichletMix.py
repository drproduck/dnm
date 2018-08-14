from matplotlib import colors
import numpy as np
from numpy.random import choice, dirichlet, choice, multinomial, permutation
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from scipy.special import psi
import copy

# mixture of exchangable edges with finite clusters using variational inference

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


def viDirichletMix(n_clusters, edges, nodes):
    """replace HDP with mixture of Dirichlet-Categorical"""
    n_edges = edges.shape[0]
    n_nodes = len(nodes)

    # store variable that will be sampled
    state = {
        'cluster_ids': [i for i in range(n_clusters)],
        'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
        # the cluster assignment of each edge
        'sizes': {s: 0 for s in range(n_clusters)},  # size of each cluster
        'beta': None, # set later

    }

    # store hyperparameters (unchanged)
    fixed = {
        'n_clusters': n_clusters,
        'pi': np.array([1]*n_clusters, dtype=float), # dirichlet prior for mixture weight
        'gamma': np.array([0.1]*len(nodes), dtype=float), # Dirichlet prior for node cluster weight
        'kappa': 0.7, # robbins-monro
        'edges': edges,
        'n_edges': n_edges,
        'n_nodes': len(nodes),
        'node_ids': None,
    }

    vi = {
        'h_c': None, # c_hat the variational param of cluster indicator c
        'h_a': None,
        'h_b': None,
        'h_alpha': None
    }

    # train parameters
    train = {
        'n_iter': 8000,
        'n_validate': 200,
        'batch_size': 100,
        'n_sample': 100,
    }

    def vi_c(ind):
        """
        update c_hat given an edge
        :return:
        """

        u = fixed['edges'][ind,0]
        v = fixed['edges'][ind,1]
        logp = []
        for k in state['cluster_ids']:
            psi_alpha = psi(vi['h_alpha'][k])
            psi_a_k_u = psi(vi['h_a'][k][u])
            psi_a_k_un = psi(sum(vi['h_a'][k]))
            psi_a_k_v = psi(vi['h_b'][k][v])
            psi_a_k_vn = psi(sum(vi['h_b'][k]))
            # print(psi_alpha, psi_a_k_u, psi_a_k_un, psi_a_k_v, psi_a_k_vn)
            logpk = psi_alpha + psi_a_k_u - psi_a_k_un + psi_a_k_v - psi_a_k_vn
            logp += [logpk]

        #underflow
        logp = logp - max(logp)
        p = np.exp(logp)
        return p / sum(p)

    def locally_collapsed_c_sample(ind):
        u = fixed['edges'][ind,0]
        v = fixed['edges'][ind,1]
        p = []
        for k in state['cluster_ids']:
            p_u_k = vi['h_a'][k][u] / sum(vi['h_a'][k])
            p_v_k = vi['h_b'][k][v] / sum(vi['h_b'][k])
            p_c_k = vi['h_alpha'][k] / sum(vi['h_alpha'])
            ps = p_u_k * p_v_k * p_c_k
            p += [ps]
        return p / sum(p)


    def cleanup():
        """
        replace viparams with empty arrays for next update
        :return:
        """
        vi['h_a'] = {i: copy.deepcopy(fixed['gamma']) for i in state['cluster_ids']}
        vi['h_b'] = {i: copy.deepcopy(fixed['gamma']) for i in state['cluster_ids']}
        vi['h_alpha'] = copy.deepcopy(fixed['pi'])

    def init():
        """
        initialize global vi parameters
        :return:
        """

        vi['h_a'] = {i: copy.deepcopy(fixed['gamma'])+np.random.rand(n_nodes) for i in state['cluster_ids']} # A_hat vi param for A each c
        vi['h_b'] = {i: copy.deepcopy(fixed['gamma'])+np.random.rand(n_nodes) for i in state['cluster_ids']} # B_hat vi param for B for each c
        vi['h_alpha'] = copy.deepcopy(fixed['pi'])+np.random.rand(n_clusters) # alpha_hat vi param for alpha

    def get_local_estimate():
        p_h_c = []
        for ind,e in enumerate(edges):
            p_ind = vi_c(ind)
            p_h_c.append(p_ind)
            assert(abs(sum(p_ind) - 1) < 0.00001)
        vi['h_c'] = p_h_c
        return vi['h_c']

    def get_batch(batch_size):

        cands = permutation(n_edges)
        i = 0
        while i < n_edges:
            res = cands[i:i+batch_size]
            yield res, len(res)
            i += batch_size

    def vi_train_stochastic(batch_size=100, locally_collapsed=False):
        print('stochastic variational inference')
        init()

        for i in range(train['n_iter']):
            print('train | iter {}'.format(i))
            # update weight
            eps = (i+2)**(-fixed['kappa'])

            for inds, batch_size_corrected in get_batch(batch_size):

                assert batch_size_corrected != 0
                p_batch = []
                for ind in inds:

                    # normal vi update
                    if not locally_collapsed:
                        p_ind = vi_c(ind)
                        p_batch.append(p_ind)
                        assert(abs(sum(p_ind) - 1) < 0.00001)

                    # locally collapsed vi update
                    else:
                        p_ind = locally_collapsed_c_sample(ind)
                        p_batch.append(p_ind)
                        assert(abs(sum(p_ind) - 1) < 0.00001)

                        # TESTING: only 1 sample
                        p_ind = multinomial(1, pvals=p_ind)
                        p_batch.append(p_ind)

                for p_ind in p_batch:
                    vi['h_alpha'] = (1-eps) * vi['h_alpha'] + eps * (fixed['pi'] + n_edges / batch_size_corrected * p_ind)

                for k in state['cluster_ids']:
                    for ind,p_ind in zip(inds,p_batch):
                        u = fixed['edges'][ind,0]
                        v = fixed['edges'][ind,1]

                        vi['h_a'][k][u] = (1-eps) * vi['h_a'][k][u] + eps * (fixed['gamma'][u] + n_edges / batch_size_corrected * p_ind[k])
                        vi['h_b'][k][v] = (1-eps) * vi['h_b'][k][v] + eps * (fixed['gamma'][v] + n_edges / batch_size_corrected * p_ind[k])

            p_h_c = get_local_estimate()
            cluster = [np.argmax(i) for i in p_h_c]
            cmap = colors.ListedColormap(['white', 'red', 'green', 'blue','yellow','purple','orange','brown','black'])
            bounds = [0,1,2,3,4,5,6,7,8,9]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            count = Counter(cluster).most_common()
            count = [c[0] for c in count]
            temp = [1,2,3,4,5,6,7,8,9]
            color = {count[x]: temp[x] if x < 8 else temp[8] for x in range(len(count))}
            sz = edges.max(0).max() + 1
            adj = np.zeros((sz,sz),dtype=int)
            for e,c in zip(edges, cluster):
                adj[e[0], e[1]] = color[c]
            plt.imshow(adj,cmap=cmap,norm=norm)
            plt.pause(1)

        plt.show()


    def vi_train():
        """main procedure"""

        print('Variational inference:')
        init()
        for i in range(train['n_iter']):
            p_h_c = []
            for ind,e in enumerate(edges):
                p_ind = vi_c(ind)
                p_h_c.append(p_ind)
                assert(abs(sum(p_ind) - 1) < 0.00001)

            cleanup()

            for ind in range(n_edges):
                vi['h_alpha'] += p_h_c[ind]
            for k in state['cluster_ids']:
                for ind in range(n_edges):
                    u = fixed['edges'][ind,0]
                    v = fixed['edges'][ind,1]

                    vi['h_a'][k][u] = vi['h_a'][k][u] + p_h_c[ind][k]
                    vi['h_b'][k][v] += p_h_c[ind][k]

            print('train | iter {}'.format(i))
            vi['h_c'] = p_h_c

            if (i % 50 == 1):
                cluster = [np.argmax(i) for i in vi['h_c']]
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

    # vi_train()

    vi_train_stochastic(100, locally_collapsed=True)
    get_local_estimate()

    return state,vi


if __name__ == '__main__':
    # edges, nodes, adj = get_data('sbm')
    # n_edges = len(edges)
    # plt.imshow(adj)
    #
    # state,vi = viDirichletMix(7, np.array(edges), np.array(nodes))
    from main_test.run import get_data_will
    links_train,links_test,clusters_train,clusters_test,nodes = get_data_will('main_test/toy_test')
    state,vi = viDirichletMix(4,np.array(links_train), nodes)
    # print(vi['h_c'])
    # print(vi['h_alpha'])
