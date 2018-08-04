from matplotlib import colors
import numpy as np
from numpy.random import choice, dirichlet
import matplotlib.pyplot as plt
import networkx as nx
from Process import Antoniak
from collections import Counter
import copy

# This is the main inference algorith mfor mixture of dirichlet networks

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


def FiniteDirichetnet(n_clusters, edges, nodes):
    """replace HDP with mixture of Dirichlet-Categorical"""
    n_edges = edges.shape[0]

    # store variable that will be sampled
    state = {
        'cluster_ids': [i for i in range(n_clusters)],
        'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
        # the cluster assignment of each edge
        'n_clusters': n_clusters,
        'sizes': {s: 0 for s in range(n_clusters)},  # size of each cluster
        'beta': None, # set later

        'in_cluster_sizes': None,
        'out_cluster_sizes': None,
    }

    # store hyperparameters (unchanged)
    fixed = {
        'alpha_D': 1, # control the number of clusters
        'gamma': np.array([0.1]*len(nodes), dtype=float), # Dirichlet prior
        'edges': edges,
        'n_edges': n_edges,
        'n_nodes': len(nodes),
        'node_ids': None,
    }

    state['in_cluster_sizes'] = {state['cluster_ids'][i]: np.zeros(fixed['n_nodes']) for i in range(state['n_clusters'])}
    state['out_cluster_sizes'] = {state['cluster_ids'][i]: np.zeros(fixed['n_nodes']) for i in range(state['n_clusters'])}
    for i,c in enumerate(state['assignments']):
        u = edges[i,0]
        v = edges[i,1]
        state['in_cluster_sizes'][c][v] += 1
        state['out_cluster_sizes'][c][u] += 1

    # update size of each cluster
    for i, c in enumerate(state['assignments']):
        state['sizes'][c] += 1

    # train parameters
    train = {
        'n_iter': 1000,
        'n_burnin': 10,
    }

    # samples from samplers i.e number of samples = number of sampling iterations
    sample = {
        'assignments': np.zeros((train['n_iter'], n_edges), dtype=int),
    }

    def x_posterior_predictive(k, ind):
        """
        sample from posterior distribution of in_cluster/out_cluster categorical
        use dirichlet-categorical posterior predictive / collapsed out the categorical distribution
        """
        u = fixed['edges'][ind,0]
        v = fixed['edges'][ind,1]
        #in_cluster
        in_cluster_k = copy.deepcopy(state['in_cluster_sizes'][k])
        if state['assignments'][ind] == k:
            in_cluster_k[v] -= 1
        assert(in_cluster_k[v] >= 0)
        in_posterior_count = fixed['gamma'] + in_cluster_k
        in_prob = in_posterior_count[v] / sum(in_posterior_count)

        #out_cluster
        out_cluster_k = copy.deepcopy(state['out_cluster_sizes'][k])
        if state['assignments'][ind] == k:
            out_cluster_k[u] -= 1
        assert(out_cluster_k[u] >= 0)
        out_posterior_count = fixed['gamma'] + out_cluster_k
        out_prob = out_posterior_count[u] / sum(out_posterior_count)
        return in_prob, out_prob

    def c_old_posterior(k, ind):
        """
        :param k: the cluster index
        :param ind: the index of current edge
        :return: p(c_n=k|c^{-n}). proportional to N
        """
        if state['assignments'][ind] == k:
            N = state['sizes'][k] - 1
        else: N = state['sizes'][k]
        return N

    def get_assignment_prob(cluster, ind):
        """
        sample from assignment conditional posterior
        p(c_n=k|u_n,v_n,c^{-n}) = p(u_n|c_n=k)p(v_n|c_n=k)p(c_n=k|c^{-n})
        """

        u = fixed['edges'][ind][0]
        v = fixed['edges'][ind][1]

        # k = new
        if cluster == -1:
            # sample a new posterior categorical for new cluster
            p_u_given_c = fixed['gamma'][u] / sum(fixed['gamma'])
            p_v_given_c = fixed['gamma'][v] / sum(fixed['gamma'])
            p_c_given_other_cs = fixed['alpha_D']
        else:
            p_u_given_c, p_v_given_c = x_posterior_predictive(cluster, ind)
            p_c_given_other_cs = c_old_posterior(cluster, ind)
        return np.log(p_u_given_c) + np.log(p_v_given_c) + np.log(p_c_given_other_cs)

    def re_cluster(new_cluster, old_cluster, ind):
        """
        update cluster assignment of edges[ind]
        including: update cluster sizes, number of clusters
        """
        if new_cluster == old_cluster:
            # same cluster, no update
            return

        u = fixed['edges'][ind,0]
        v = fixed['edges'][ind,1]

        if new_cluster == -1:
            state['n_clusters'] += 1
            # create new cluster id, find the smallest number not in current cluster ids
            sorted_ids = sorted(state['cluster_ids'])
            if sorted_ids[0] > 0: new_cluster = 0
            else:
                sorted_ids += [sorted_ids[-1]+2]
                for id in range(len(sorted_ids)-1):
                    if sorted_ids[id] + 1 < sorted_ids[id+1]:
                        new_cluster = sorted_ids[id] + 1
                        break
            state['cluster_ids'] += [new_cluster]
            state['assignments'][ind] = new_cluster

            # update size
            state['sizes'][new_cluster] = 1
            state['sizes'][old_cluster] -= 1

            state['in_cluster_sizes'][new_cluster] = np.zeros(fixed['n_nodes'])
            state['in_cluster_sizes'][new_cluster][v] += 1
            state['out_cluster_sizes'][new_cluster] = np.zeros(fixed['n_nodes'])
            state['out_cluster_sizes'][new_cluster][u] += 1
            state['in_cluster_sizes'][old_cluster][v] -= 1
            state['out_cluster_sizes'][old_cluster][u] -= 1

        else:
            # update
            state['assignments'][ind] = new_cluster

            # update size
            state['sizes'][new_cluster] += 1
            state['sizes'][old_cluster] -= 1
            state['in_cluster_sizes'][new_cluster][v] += 1
            state['in_cluster_sizes'][old_cluster][v] -= 1
            state['out_cluster_sizes'][new_cluster][u] += 1
            state['out_cluster_sizes'][old_cluster][u] -= 1

        # check if cluster needs prune
        if state['sizes'][old_cluster] == 0:
            # print('old cluster removed')
            state['n_clusters'] -= 1

            state['sizes'].pop(old_cluster, None)
            state['cluster_ids'].remove(old_cluster)
            state['in_cluster_sizes'].pop(old_cluster, None)
            state['out_cluster_sizes'].pop(old_cluster, None)

    def sample_assignment(ind):
        """
        step 5
        sample new cluster assignment
        """

        old_assignment = state['assignments'][ind]
        assignment_probs = [get_assignment_prob(c, ind) for c in (state['cluster_ids'] + [-1])]

        # to reduce underflow
        assignment_probs = np.array(assignment_probs) - max(assignment_probs)
        assignment_probs = np.exp(assignment_probs)
        assignment_probs /= sum(assignment_probs)
        # print('assignment probs:',assignment_probs)

        new_assignment = choice(state['cluster_ids'] + [-1], p=assignment_probs)

        re_cluster(new_assignment, old_assignment, ind)


    def gibbs_step():
        """1 step of Gibbs, sample for each edge randomly."""

        for ind in np.random.permutation(n_edges):
            sample_assignment(ind) # sample c

    def gibbs():
        """main procedure"""

        print('Dirichlet Process gibbs sampler')

        for i in range(train['n_burnin']):
            gibbs_step()
            print('burn in | iter {}, number of clusters {}\n{}'.format(i, state['n_clusters'], state['assignments']))

        for i in range(train['n_iter']):
            gibbs_step()
            sample['assignments'][i, :] = state['assignments']

            print('train | iter {}, number of clusters {}, cluster indices {}\n{}'.format(i, state['n_clusters'], state['cluster_ids'], state['assignments'], ))
            # print('beta',state['beta'])

            # check if number of clusters is correct
            try: state['n_clusters'] == len(set(state['assignments']))
            except:
                print('wrong number of clusters')
                print(state['n_clusters'])
                print(len(set(state['assignments'])))
                exit(1)

            plt.clf()
            plt.subplot(121)

            cmap = colors.ListedColormap(['white', 'red', 'green', 'blue','yellow'])
            bounds = [0,1,2,3,4,5]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            #
            # nx.draw_networkx_nodes(G, pos, node_size=1)
            # nx.draw_networkx_edges(G, pos, edgelist=edge_tuple, edge_color=state['assignments'])
            # plt.xlim(0, 1)
            # plt.ylim(-1, 7)
            # plt.subplot(122)
            # adj = np.zeros((n_nodes, n_nodes), dtype=int)

            # color the 3 biggest clusters + 1 color for the rest
            count = Counter(state['assignments']).most_common()
            print(count)
            count = [c[0] for c in count]
            print(count)
            temp = [1,2,3,4,5]
            color = {count[x]: temp[x] if x < 3 else temp[3] for x in range(len(count))}
            for e,c in zip(edges, state['assignments']):
                adj[e[0], e[1]] = color[c]
            plt.imshow(adj,cmap=cmap,norm=norm)

            plt.pause(1)

        plt.show()

    gibbs()

    return state,sample,train


if __name__ == '__main__':
    edges, nodes, adj = get_data('sbm')
    n_edges = len(edges)
    state,sample,train = FiniteDirichetnet(2, np.array(edges), np.array(nodes))
    sim = np.zeros([n_edges, n_edges], dtype=int)
    for i in range(n_edges):
        for j in range(n_edges):
            for k in range(train['n_iter']):
                if sample['assignments'][k][i] == sample['assignments'][k][j]:
                    sim[i,j] += 1
    print(sim)
    top = np.zeros([n_edges,3])
    for i,row in enumerate(sim):
        row[i] = 0
        a = np.argsort(row)
        top[i][:3] = a[-3:]
    print(top)
    plt.show()

