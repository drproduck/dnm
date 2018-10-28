from utils import *
from matplotlib import colors
import numpy as np
from numpy.random import choice, dirichlet, choice, multinomial, permutation, gamma
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import copy
from data.get_data import display_adjacency, get_data_will

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

class Params(object):
    def __init__(self,alpha,tau,**kwargs):
        self.alpha = alpha
        self.tau = tau
        for key,value in kwargs.items():
            setattr(self, key, value)

class sequentialDN():
    """
    finite nodes infinite clusters Dirichlet Networks, with Li's sequential learning algorithm
    """

    def __init__(self, edges, nodes, edges_test=None,**kwargs):
        self.edges = edges
        self.n_edges = edges.shape[0]
        self.nodes = nodes
        self.n_nodes = nodes.shape[0]
        if edges_test is not None:
            self.edges_test = edges_test
        # hyper_params = Params(**kwargs)

        # store variable that will be sampled
        self.state = {
            'n_clusters': 0, # can change!
            'next_cluster': 0,
            # 'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
            # the cluster assignment of each edge
            # 'sizes': Counter({i: 0 for i in range(n_clusters)}),  # size of each cluster
            'assignments': np.zeros(self.n_edges),
            'cluster_ids': [],
            'log_ll': [] # heldout loglikelihood for each iteration
        }

        self.tau = 1
        # store hyperparameters (unchanged)
        self.fixed = {
            'tau': self.tau, # prior for Dirichlet of node weight
            'alpha': 1, # Beta prior for cluster stick breaking ~ Beta(1, \alpha)
            'lambda1': np.array([self.tau/self.n_nodes]*len(nodes), dtype=float), # Dirichlet prior for node
            'lambda2': None, # Dirichlet prior total count
            'edges': edges,
            'n_edges': edges.shape[0],
            'n_nodes': nodes.shape[0],
        }

        self.fixed['lambda2'] = self.fixed['lambda1'].sum()

        self.vi = {
            'zeta1': {}, # pseudo count of outnodes
            'zeta2': {}, # pseudo count of innodes
            'zeta_sum1': {}, # total count of outnodes
            'zeta_sum2': {}, # total count of innodes

            'w': {}, # weight for each cluster
            'rho': {}, # assignment posterior for each edge
            'rho_sum': {}, # caching sum for normalization
        }

        # self.train parameters
        self.train = {
            'n_iter': 1,
            'batch_size': 100,
            'n_sample': 100,
            'eps': 0.5, # new cluster threshold
            'eps_r': 0.01, # prune
            'eps_d': 0, # merge
        }

    def marginal_ll(self, ind, cluster):
        u = self.edges[ind][0]
        v = self.edges[ind][1]

        #new cluster
        if cluster == -1:
            lu = np.log(self.fixed['lambda1'][u]) - np.log(self.fixed['lambda2'])
            lv = np.log(self.fixed['lambda1'][v]) - np.log(self.fixed['lambda2'])
            return lu + lv
        else:
            lu = np.log(self.vi['zeta1'][cluster][u]) - np.log(self.vi['zeta_sum1'][cluster])
            lv = np.log(self.vi['zeta2'][cluster][v]) - np.log(self.vi['zeta_sum2'][cluster])
            return lu + lv

    def rho(self, likelihood):
        """
        assignment log posterior (unnormalized)
        :param likelihood:
        :return:
        """
        p = []
        for i,c in enumerate(self.state['cluster_ids']):
            p += [np.log(self.vi['w'][c]) + likelihood[i]]
        p += [np.log(self.fixed['alpha']) + likelihood[-1]] # new cluster
        return p

    def get_map(self):
        cluster = []
        for ind in range(self.n_edges):
            hi = np.array([self.marginal_ll(ind, cluster) for cluster in self.state['cluster_ids']])
            p = []
            for i,c in enumerate(self.state['cluster_ids']):
                p += [np.log(self.vi['w'][c]) + hi[i]]
            cluster += [self.state['cluster_ids'][np.argmax(p)]]

        return cluster

    def get_batch(self,batch_size):
        i = 0
        cands = permutation(self.n_edges)
        while i < self.n_edges:
            res = cands[i: i+batch_size]
            yield res, len(res)
            i += batch_size

    def add_new_cluster(self):
        """
        add new cluster
        :return:
        """

        new_cluster = self.state['next_cluster']
        self.state['cluster_ids'] += [new_cluster]
        self.state['next_cluster'] += 1
        self.state['n_clusters'] += 1
        self.vi['zeta1'][new_cluster] = np.zeros(self.n_nodes) + self.fixed['tau'] / self.n_nodes
        self.vi['zeta2'][new_cluster] = np.zeros(self.n_nodes) + self.fixed['tau'] / self.n_nodes
        self.vi['zeta_sum1'][new_cluster] = 0 + self.fixed['tau']
        self.vi['zeta_sum2'][new_cluster] = 0 + self.fixed['tau']
        # self.vi['zeta1'][new_cluster] = copy.deepcopy(self.fixed['lambda1'])
        # self.vi['zeta2'][new_cluster] = copy.deepcopy(self.fixed['lambda1'])
        # self.vi['zeta_sum1'][new_cluster] = copy.deepcopy(self.fixed['lambda2'])
        # self.vi['zeta_sum2'][new_cluster] = copy.deepcopy(self.fixed['lambda2'])
        # self.vi['zeta1'][new_cluster] = np.zeros(n_nodes)
        # self.vi['zeta2'][new_cluster] = np.zeros(n_nodes)
        # self.vi['zeta_sum1'][new_cluster] = 0
        # self.vi['zeta_sum2'][new_cluster] = 0
        self.vi['w'][new_cluster] = self.fixed['alpha']

    def heldout_loglikelihood(self,edges_test):
        """
        compute mean of parameters
        :return:
        """
        self.vi['e_zeta1'] = {}
        self.vi['e_zeta2'] = {}
        self.vi['e_w'] = {}
        for cluster in self.state['cluster_ids']:
            assert(abs(self.vi['zeta1'][cluster].sum() - self.vi['zeta_sum1'][cluster]) < 0.000001)
            assert(abs(self.vi['zeta2'][cluster].sum() - self.vi['zeta_sum2'][cluster]) < 0.000001)
            self.vi['e_zeta1'][cluster] = self.vi['zeta1'][cluster] / self.vi['zeta_sum1'][cluster]
            self.vi['e_zeta2'][cluster] = self.vi['zeta2'][cluster] / self.vi['zeta_sum2'][cluster]
        w_sum = sum([self.vi['w'][i] for i in self.state['cluster_ids']])
        for cluster in self.state['cluster_ids']:
            self.vi['e_w'][cluster] = self.vi['w'][cluster] / w_sum
        ll_sum = 0
        for e in edges_test:
            u = e[0]
            v = e[1]
            lp = 0
            for cluster in self.state['cluster_ids']:
                # print(self.vi['e_w'][cluster], self.vi['e_zeta1'][cluster][u], self.vi['e_zeta2'][cluster][v])
                lp += self.vi['e_w'][cluster] * self.vi['e_zeta1'][cluster][u] * self.vi['e_zeta2'][cluster][v]

                if np.isneginf(np.log(lp)):
                    print(self.vi['e_w'][cluster], self.vi['e_zeta1'][cluster][u], self.vi['e_zeta2'][cluster][v])
                    exit()

            ll_sum += np.log(lp)
        assert(not np.isneginf(ll_sum))
        return ll_sum / len(edges_test)

    def merge(self,merger, merged):
        """
        function to merge 2 groups
        :param merger:
        :param merged:
        :return:
        """
        if merger == merged:
            return None
        self.vi['zeta1'][merger] = self.vi['zeta1'][merger] + self.vi['zeta1'][merged]
        self.vi['zeta2'][merger] = self.vi['zeta2'][merger] + self.vi['zeta2'][merged]
        self.vi['zeta_sum1'][merger] += self.vi['zeta_sum1'][merged]
        self.vi['zeta_sum2'][merger] += self.vi['zeta_sum2'][merged]
        self.vi['w'][merger] += self.vi['w'][merged]

        self.prune(merged)
        return merged

    def prune(self, pruned):

        self.state['cluster_ids'].remove(pruned)
        self.vi['zeta1'].pop(pruned, None)
        self.vi['zeta2'].pop(pruned, None)
        self.vi['zeta_sum1'].pop(pruned, None)
        self.vi['zeta_sum2'].pop(pruned, None)
        self.vi['w'].pop(pruned, None)
        self.state['n_clusters'] -= 1
        return pruned

    def rho_adjust(self, removed, n_cur_edges):
        for ind in range(n_cur_edges):
            for r in removed:
                if r in self.vi['rho'][ind]: # sometimes its not in rho because the cluster is created later
                    self.vi['rho_sum'][ind] -= self.vi['rho'][ind].pop(r)

    def merge_large(self, n_cur_edges):
        sets = UnionFind()

        n_sc = []
        for i in self.state['cluster_ids']:
            for j in self.state['cluster_ids']:
                if i != j:
                    sc = 0
                    for ind in range(n_cur_edges):
                        if i in self.vi['rho'][ind]:
                            pi = self.vi['rho'][ind][i] / self.vi['rho_sum'][ind]
                        else:
                            pi = 0
                        if j in self.vi['rho'][ind]:
                            pj = self.vi['rho'][ind][j] /self.vi['rho_sum'][ind]
                        else:
                            pj = 0
                        sc += 1 / (ind+1) * abs(pi - pj)
                    if sc < self.train['eps_d']:
                        sets.union(i, j)
                    n_sc += [sc]

        print('average score by Lin ',np.array(n_sc).mean())
        removed = []
        for key, value in sets.parent_pointers.items():
            merged = sets.num_to_objects[key]
            merger = sets.num_to_objects[value]
            removing = self.merge(merger, merged)
            if removing is not None:
                removed += [removing]

        self.rho_adjust(removed, n_cur_edges)

        return removed

    def prune_large(self, n_cur_edges):
        cluster = []
        weight = []
        for x in self.vi['w'].items():
            cluster += [x[0]]
            weight += [x[1]]
        weight = weight / sum(weight)

        pruned = []
        for c, w in zip(cluster, weight):
            if w < self.train['eps_r']:
                pr = self.prune(c)
                if pr is not None:
                    pruned += [pr]
        self.rho_adjust(pruned, n_cur_edges)

        return pruned

    def generate(self, n):
        """

        :return:
        """
        edges_gen = []
        ids = []
        w = []
        assert(len(self.state['cluster_ids']) == len(self.vi['e_w']))
        for k,v in self.vi['e_w'].items():
            ids += [k]
            w += [v]
        assert(abs(sum(w) - 1) < 0.000001)
        clusters = choice(ids, size=n, replace=True, p=w)
        for cluster in clusters:
            a = choice(self.n_nodes, p=self.vi['e_zeta1'][cluster], replace=True)
            b = choice(self.n_nodes, p=self.vi['e_zeta2'][cluster], replace=True)
            edges_gen.append([a,b])

        return np.array(edges_gen), clusters

    def vi_train_stochastic(self):
        """
        update lambda, optimal ordering, then update stick
        :param batch_size:
        :return:
        """
        print('sequential variational inference')
        # init_self.vi()

        for iter in range(self.train['n_iter']):

            # first point first cluster
            if iter == 0:
                self.add_new_cluster()
                u = self.edges[0,0]
                v = self.edges[0,1]
                self.vi['w'][0] = 1
                self.vi['rho'][0] = {0: 1}
                self.vi['rho_sum'][0] = 1
                self.vi['zeta1'][0][u] += 1
                self.vi['zeta_sum1'][0] += 1
                self.vi['zeta2'][0][v] += 1
                self.vi['zeta_sum2'][0] += 1

            for ind in range(1,self.n_edges):
                u = self.edges[ind,0]
                v = self.edges[ind,1]
                h = [self.marginal_ll(ind, cluster) for cluster in self.state['cluster_ids'] + [-1]]
                unnormed_p = np.array(self.rho(h))
                lp,_ = log_normalize(unnormed_p)
                p = np.exp(lp)

                if p[-1] > self.train['eps']:
                    self.add_new_cluster()

                elif p[-1] <= self.train['eps']:
                    unnormed_p = unnormed_p[:-1]
                    lp,_ = log_normalize(unnormed_p)
                    p = np.exp(lp)

                # set rho here
                self.vi['rho'][ind] = {c: p[i] for i,c in enumerate(self.state['cluster_ids'])}
                self.vi['rho_sum'][ind] = 1

                assert(abs(p.sum() - 1) < 0.00001)

                for cluster in self.state['cluster_ids']:

                    self.vi['w'][cluster] += self.vi['rho'][ind][cluster]
                    self.vi['zeta1'][cluster][u] += self.vi['rho'][ind][cluster]
                    self.vi['zeta2'][cluster][v] += self.vi['rho'][ind][cluster]
                    self.vi['zeta_sum2'][cluster] += self.vi['rho'][ind][cluster]
                    self.vi['zeta_sum1'][cluster] += self.vi['rho'][ind][cluster]

                    # self.vi['zeta_sum2'][cluster] += 1
                    # self.vi['zeta_sum1'][cluster] += 1

                    assert(abs(self.vi['zeta1'][cluster].sum() - self.vi['zeta_sum1'][cluster]) < 0.000001)
                    assert(abs(self.vi['zeta2'][cluster].sum() - self.vi['zeta_sum2'][cluster]) < 0.000001)

                if ind % 50 == 0:
                    self.state['log_ll'] += [self.heldout_loglikelihood(self.edges_test)]
                if ind % 200 == 0 or ind == self.n_edges - 1:

                    # merge
                    merged = self.merge_large(ind+1)
                    # prune
                    pruned = self.prune_large(ind+1)

                    print('merged ',len(merged))
                    print('pruned ',len(pruned))

                    print(self.state['n_clusters'])

                    # CHECK
                    for ind in range(ind+1):
                        x = sum([i for i in self.vi['rho'][ind].values()])
                        assert(abs(x - self.vi['rho_sum'][ind]) < 0.00001)

            self.state['log_ll'] += [self.heldout_loglikelihood(self.edges_test)] # 1 last time

            assert(self.state['n_clusters'] == len(self.state['cluster_ids']))
            print('self.train | iter {}'.format(iter))
            cluster = self.get_map()
            cmap = colors.ListedColormap(['white', 'red', 'green', 'blue','yellow','purple','orange','brown','black'])
            bounds = [0,1,2,3,4,5,6,7,8,9]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            count = Counter(cluster).most_common()
            count = [c[0] for c in count]
            temp = [1,2,3,4,5,6,7,8,9]
            color = {count[x]: temp[x] if x < 8 else temp[8] for x in range(len(count))}
            sz = self.edges.max(axis=1).max() + 1
            adj = np.zeros([sz,sz],dtype=int)
            for e,c in zip(self.edges, cluster):
                # adj[e[0], e[1]] = c
                adj[e[0], e[1]] = color[c]
            plt.imshow(adj,cmap=cmap,norm=norm)
            # plt.imshow(adj)

            plt.pause(1)

        plt.figure(1)
        print(self.state['log_ll'])
        plt.plot(self.state['log_ll'])
        plt.show()


if __name__ == '__main__':
    links_train,links_test,clusters_train,clusters_test,nodes,node_clusters = get_data_will('toy_test',ratio=0.9)
    DN = sequentialDN(links_train, np.array(nodes), links_test)
    DN.vi_train_stochastic()
    # edges, nodes, adj = get_data('sbm')
    # n_edges = len(edges)
    # plt.imshow(adj)
    #
    # self.state,self.vi = sequentialDN(np.array(edges), np.array(nodes))
    edges, clusters = DN.generate(100)
    display_adjacency(edges, clusters)








