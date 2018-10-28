from utils import *
from matplotlib import colors
import numpy as np
from numpy.random import choice, dirichlet, choice, multinomial, permutation, gamma
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
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


def sequentialDN(edges, nodes,edges_test):
    """
    finite nodes infinite clusters Dirichlet Networks, with Li's sequential learning algorithm
    :param edges:
    :param nodes:
    :return:
    """
    n_edges = edges.shape[0]
    n_nodes = len(nodes)

    # store variable that will be sampled
    state = {
        'n_clusters': 0, # can change!
        'next_cluster': 0,
        # 'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
        # the cluster assignment of each edge
        # 'sizes': Counter({i: 0 for i in range(n_clusters)}),  # size of each cluster
        'assignments': np.zeros(n_edges),
        'cluster_ids': [],
        'log_ll': [] # heldout loglikelihood for each iteration
    }

    # store hyperparameters (unchanged)
    fixed = {
        'tau': 3, # prior for Dirichlet of node weight
        'alpha': 3, # Beta prior for cluster stick breaking ~ Beta(1, \alpha)
        'lambda1': np.array([1/n_nodes]*len(nodes), dtype=float), # Dirichlet prior for node

        'lambda2': None, # Dirichlet prior total count
        'edges': edges,
        'n_edges': n_edges,
        'n_nodes': len(nodes),
    }

    fixed['lambda2'] = fixed['lambda1'].sum()

    vi = {
        'zeta1': {}, # pseudo count of outnodes
        'zeta2': {}, # pseudo count of innodes
        'zeta_sum1': {}, # total count of outnodes
        'zeta_sum2': {}, # total count of innodes

        'w': {}, # weight for each cluster
        'rho': {}, # assignment posterior for each edge
    }

    # train parameters
    train = {
        'n_iter': 1,
        'batch_size': 100,
        'n_sample': 100,
        'eps': 0.5, # new cluster threshold
        'eps_r': 0.01, # prune
        'eps_d': 0.1, # merge
    }

    def marginal_ll(ind, cluster):
        u = edges[ind][0]
        v = edges[ind][1]

        #new cluster
        if cluster == -1:
            lu = np.log(fixed['lambda1'][u]) - np.log(fixed['lambda2'])
            lv = np.log(fixed['lambda1'][v]) - np.log(fixed['lambda2'])
            return lu + lv
        else:
            lu = np.log(vi['zeta1'][cluster][u]) - np.log(vi['zeta_sum1'][cluster])
            lv = np.log(vi['zeta2'][cluster][v]) - np.log(vi['zeta_sum2'][cluster])
            return lu + lv

    def rho(ll):
        """
        assignment log posterior (unnormalized)
        :param ll:
        :return:
        """
        p = []
        for i,c in enumerate(state['cluster_ids']):
            p += [np.log(vi['w'][c]) + ll[i]]
        p += [np.log(fixed['alpha']) + ll[-1]] # new cluster
        return p

    def get_map():
        cluster = []
        for ind in range(n_edges):
            hi = np.array([marginal_ll(ind, cluster) for cluster in state['cluster_ids']])
            p = []
            for i,c in enumerate(state['cluster_ids']):
                p += [np.log(vi['w'][c]) + hi[i]]
            # print(lp)
            cluster += [state['cluster_ids'][np.argmax(p)]]

        return cluster

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

        new_cluster = state['next_cluster']
        state['cluster_ids'] += [new_cluster]
        state['next_cluster'] += 1
        state['n_clusters'] += 1
        vi['zeta1'][new_cluster] = np.zeros(n_nodes) + 1 / n_nodes
        vi['zeta2'][new_cluster] = np.zeros(n_nodes) + 1 / n_nodes
        vi['zeta_sum1'][new_cluster] = 1
        vi['zeta_sum2'][new_cluster] = 1
        # vi['zeta1'][new_cluster] = copy.deepcopy(fixed['lambda1'])
        # vi['zeta2'][new_cluster] = copy.deepcopy(fixed['lambda1'])
        # vi['zeta_sum1'][new_cluster] = copy.deepcopy(fixed['lambda2'])
        # vi['zeta_sum2'][new_cluster] = copy.deepcopy(fixed['lambda2'])
        # vi['zeta1'][new_cluster] = np.zeros(n_nodes)
        # vi['zeta2'][new_cluster] = np.zeros(n_nodes)
        # vi['zeta_sum1'][new_cluster] = 0
        # vi['zeta_sum2'][new_cluster] = 0
        vi['w'][new_cluster] = fixed['alpha']

    def heldout_loglikelihood(edges_test):
        """
        compute mean of parameters
        :return:
        """
        vi['e_zeta1'] = {}
        vi['e_zeta2'] = {}
        vi['e_w'] = {}
        for cluster in state['cluster_ids']:
            # assert(abs(vi['zeta1'][cluster].sum() - vi['zeta_sum1'][cluster]) < 0.000001)
            # assert(abs(vi['zeta2'][cluster].sum() - vi['zeta_sum2'][cluster]) < 0.000001)
            vi['e_zeta1'][cluster] = vi['zeta1'][cluster] / vi['zeta_sum1'][cluster]
            vi['e_zeta2'][cluster] = vi['zeta2'][cluster] / vi['zeta_sum2'][cluster]
        w_sum = sum([vi['w'][i] for i in state['cluster_ids']])
        for cluster in state['cluster_ids']:
            vi['e_w'][cluster] = vi['w'][cluster] / w_sum
        ll_sum = 0
        for e in edges_test:
            u = e[0]
            v = e[1]
            lp = 0
            for cluster in state['cluster_ids']:
                # print(vi['e_w'][cluster], vi['e_zeta1'][cluster][u], vi['e_zeta2'][cluster][v])
                lp += vi['e_w'][cluster] * vi['e_zeta1'][cluster][u] * vi['e_zeta2'][cluster][v]

                if np.isneginf(np.log(lp)):
                    print(vi['e_w'][cluster], vi['e_zeta1'][cluster][u], vi['e_zeta2'][cluster][v])
                    exit()

            print(lp)
            ll_sum += np.log(lp)
            print(ll_sum)
        assert(not np.isneginf(ll_sum))
        print(len(edges_test))
        return ll_sum / len(edges_test)

    def merge(merger, merged):
        """
        function to merge 2 groups
        :param merger:
        :param merged:
        :return:
        """
        vi['zeta1'][merger] = vi['zeta1'][merger] + vi['zeta1'][merged]
        vi['zeta2'][merger] = vi['zeta2'][merger] + vi['zeta2'][merged]
        vi['zeta_sum1'][merger] += vi['zeta_sum1'][merged]
        vi['zeta_sum2'][merger] += vi['zeta_sum2'][merged]
        vi['w'][merger] += vi['w'][merged]

        state['cluster_ids'].remove(merged)
        vi['zeta1'].pop(merged, None)
        vi['zeta2'].pop(merged, None)
        vi['zeta_sum1'].pop(merged, None)
        vi['zeta_sum2'].pop(merged, None)
        vi['w'].pop(merged, None)
        state['n_clusters'] -= 1

    def vi_train_stochastic():
        """
        update lambda, optimal ordering, then update stick
        :param batch_size:
        :return:
        """
        print('sequential variational inference')
        # init_vi()

        for iter in range(train['n_iter']):

            # first point first cluster
            add_new_cluster()
            u = edges[0,0]
            v = edges[0,1]
            vi['w'][0] = 1
            vi['rho'][0] = {0: 1}
            vi['zeta1'][0][u] += 1
            vi['zeta_sum1'][0] += 1
            vi['zeta2'][0][v] += 1
            vi['zeta_sum2'][0] += 1

            for ind in range(1,n_edges):
                u = edges[ind,0]
                v = edges[ind,1]
                h = [marginal_ll(ind, cluster) for cluster in state['cluster_ids'] + [-1]]
                unnormed_p = np.array(rho(h))
                lp,_ = log_normalize(unnormed_p)
                p = np.exp(lp)

                if p[-1] > train['eps']:
                    add_new_cluster()

                elif p[-1] <= train['eps']:
                    unnormed_p = unnormed_p[:-1]
                    lp,_ = log_normalize(unnormed_p)
                    p = np.exp(lp)

                vi['rho'][ind] = {c: p[i] for i,c in enumerate(state['cluster_ids'])}

                assert(abs(p.sum() - 1) < 0.00001)

                for cluster in state['cluster_ids']:
                    vi['w'][cluster] += vi['rho'][ind][cluster]
                    vi['zeta1'][cluster][u] += vi['rho'][ind][cluster]
                    vi['zeta2'][cluster][v] += vi['rho'][ind][cluster]
                    vi['zeta_sum2'][cluster] += vi['rho'][ind][cluster]
                    vi['zeta_sum1'][cluster] += vi['rho'][ind][cluster]

                    # vi['zeta_sum2'][cluster] += 1
                    # vi['zeta_sum1'][cluster] += 1

                    # assert(abs(vi['zeta1'][cluster].sum() - vi['zeta_sum1'][cluster]) < 0.000001)
                    # assert(abs(vi['zeta2'][cluster].sum() - vi['zeta_sum2'][cluster]) < 0.000001)
                # print(state['n_clusters'])
                if ind % 50 == 0 or ind == n_edges - 1:
                    # merge
                    # for i in state['cluster_ids']:
                    #     for j in state['cluster_ids']:
                    #         if i != j:
                    #             sc = 0
                    #             for ind in range(n_edges):
                    #                 if i in vi['rho'][ind]:
                    #                     pi = vi['rho'][ind][i]
                    #                 else:
                    #                     pi = 0
                    #                 if j in vi['rho'][ind]:
                    #                     pj = vi['rho'][ind][j]
                    #                 else:
                    #                     pj = 0
                    #                 sc += 1 / (ind+1) * abs(pi - pj)
                    #             if sc < train['eps_d']:
                    #                 # merge


                    # prune
                    # weight = [i for i in vi['w'].values()]
                    # cluster = [i for i in vi['w'].keys()]
                    cluster = []
                    weight = []
                    for x in vi['w'].items():
                        cluster += [x[0]]
                        weight += [x[1]]
                    weight = weight / sum(weight)
                    for c, w in zip(cluster, weight):
                        if w < train['eps_r']:
                            # del vi['rho'][i]
                            state['cluster_ids'].remove(c)
                            vi['zeta1'].pop(c, None)
                            vi['zeta2'].pop(c, None)
                            vi['zeta_sum1'].pop(c, None)
                            vi['zeta_sum2'].pop(c, None)
                            vi['w'].pop(c, None)
                            state['n_clusters'] -= 1

                    state['log_ll'] += [heldout_loglikelihood(edges_test)]

            state['log_ll'] += [heldout_loglikelihood(edges_test)] # 1 last time

            assert(state['n_clusters'] == len(state['cluster_ids']))
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
            sz = edges.max(axis=1).max() + 1
            adj = np.zeros([sz,sz],dtype=int)
            for e,c in zip(edges, cluster):
                adj[e[0], e[1]] = color[c]
            plt.imshow(adj,cmap=cmap,norm=norm)

            plt.pause(1)

        plt.figure(1)
        print(state['log_ll'])
        plt.plot(state['log_ll'])
        plt.show()
    vi_train_stochastic()

    return state,vi


if __name__ == '__main__':
    from main_test.run import get_data
    links_train,links_test,clusters_train,clusters_test,nodes = get_data('csli_presentation/toy_test')
    state,vi = sequentialDN(links_train, nodes, links_test)
    # print(vi['h_c'])
    # print(vi['h_alpha'])
    # edges, nodes, adj = get_data('sbm')
    # n_edges = len(edges)
    # plt.imshow(adj)
    #
    # state,vi = sequentialDN(np.array(edges), np.array(nodes))
