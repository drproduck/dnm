import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
import networkx as nx


def get_data(fname):
    """should return an (n,2) array of edges"""
    f = open(fname, 'r')
    res = np.array([list(map(int, line.strip().split(' '))) for line in f], dtype=int)
    sz = int(res.flatten().max() + 1)
    adj = np.zeros((sz, sz), dtype=int)

    for e in res:
        adj[e[0], e[1]] += 1

    return res, adj


def mdnd(n_clusters, edges):
    n_edges = edges.shape[0]
    state = {
        'cluster_ids': [i for i in range(n_clusters)],  # ?
        'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
        # the cluster assignment of each edge
        'n_clusters': n_clusters,
        'sizes': {s: 0 for s in range(n_clusters)}  # size of each cluster
    }

    fixed = {
        'alpha_D': 10, # control the number of clusters
        'tau': 5, # control cluster overlap
        'gamma_H': 5, # control number of nodes
        'edges': edges,
        'n_edges': n_edges,
        # 'node_indices': set(),
        'node_weights': dict(),

        # update these, else very slow
        'inlinks': dict(),
        'outlinks': dict()
    }

    train = {
        'n_iter': 5,
        'n_burnin': 1,
    }

    # samples from samplers i.e number of samples = number of sampling iterations
    sample = {
        'assignments': np.zeros((train['n_iter'], n_edges), dtype=int),
    }

    # the importance beta in the paper, as node weights normalized to sum 1
    sw = 0
    for e in edges:
        if e[0] not in fixed['node_weights']:
            fixed['node_weights'][e[0]] = 1
        else:
            fixed['node_weights'][e[0]] += 1
        sw += 1

        if e[1] != e[0]:  # avoid double count self-edge
            sw += 1
            if e[1] not in fixed['node_weights']:
                fixed['node_weights'][e[1]] = 1
            else:
                fixed['node_weights'][e[1]] += 1

    # normalize
    for k in fixed['node_weights'].keys():
        fixed['node_weights'][k] /= sw

    # TODO: collect inlinks and outlinks (updatable), currently look through edges at each step (slow)
    def get_inlink_size(v, k, ind):
        """return: # inlinks of v associated with cluster k"""
        res = [a and b for a, b in zip(fixed['edges'][:, 1] == v, state['assignments'] == k)]
        # res = [fixed['edges'][:,1] == v and state['assignments'] == k]

        # VERY slow check
        # res1 = 0
        # for i,e in enumerate(edges):
        #     if state['assignments'][i] == k and e[1] == v:
        #         res1 += 1
        # assert(sum(res) == res1)

        # remove edge[ind] if its in inlink
        res[ind] = False
        return sum(res)

    def get_outlink_size(u, k, ind):
        res = [a and b for a, b in zip(fixed['edges'][:, 0] == u, state['assignments'] == k)]
        # res = [fixed['edges'][:,0] == u and state['assignments'] == k]

        # VERY slow check
        # res1 = 0
        # for i,e in enumerate(edges):
        #     if state['assignments'][i] == k and e[0] == u:
        #         res1 += 1
        # assert(sum(res) == res1)
        res[ind] = False
        return sum(res)

    def get_prob(cluster, ind):
        u = fixed['edges'][ind][0]
        v = fixed['edges'][ind][1]

        # new cluster
        if cluster == -1:
            return np.log(fixed['alpha_D']) + 2 * np.log(fixed['tau']) + np.log(fixed['node_weights'][u]) + np.log(fixed['node_weights'][v])

        else:
            if state['assignments'][ind] == cluster:
                Nk = state['sizes'][cluster] - 1
            else:
                Nk = state['sizes'][cluster]

            Lu = get_outlink_size(u, cluster, ind)
            Lv = get_inlink_size(v, cluster, ind)

            return np.log(Nk) + np.log(Lu + fixed['tau'] * fixed['node_weights'][u]) + np.log(Lv + fixed['tau'] * fixed['node_weights'][v])

    def re_cluster(new_cluster, old_cluster, ind):
        """
        update cluster assignment of edges[ind]
        including: update cluster sizes, number of clusters
        """
        if new_cluster == old_cluster:
            # same cluster, no update
            return

        if new_cluster == -1:
            state['n_clusters'] += 1
            # create new cluster id
            new_cluster = max(state['cluster_ids']) + 1
            state['cluster_ids'] += [new_cluster]
            state['assignments'][ind] = new_cluster

            # update size
            state['sizes'][new_cluster] = 1
            state['sizes'][old_cluster] -= 1

        else:
            # update
            state['assignments'][ind] = new_cluster

            # update size
            state['sizes'][new_cluster] += 1
            state['sizes'][old_cluster] -= 1

        # check if cluster needs prune
        if state['sizes'][old_cluster] == 0:
            # print('old cluster removed')
            state['n_clusters'] -= 1

            state['sizes'].pop(old_cluster, None)
            state['cluster_ids'].remove(old_cluster)

    def assignment_posterior(ind):
        """sample new cluster assignment"""

        old_assignment = state['assignments'][ind]
        assignment_probs = [get_prob(c, ind) for c in (state['cluster_ids'] + [-1])]

        # to reduce underflow
        assignment_probs = assignment_probs - max(assignment_probs)
        assignment_probs = np.exp(assignment_probs)

        new_assignment = choice(state['cluster_ids'] + [-1], p=assignment_probs / sum(assignment_probs))

        re_cluster(new_assignment, old_assignment, ind)

    def gibbs_step():
        """1 step of Gibbs, sample for each edge randomly."""

        for ind in np.random.permutation(n_edges):
            assignment_posterior(ind)

    def gibbs():
        """main procedure"""

        print('Dirichlet Process gibbs sampler')

        # update and size of each cluster
        for i, c in enumerate(state['assignments']):
            state['sizes'][c] += 1

        for i in range(train['n_burnin']):
            gibbs_step()
            print('burn in | iter {}, number of clusters {}\n{}'.format(i, state['n_clusters'], state['assignments']))

        for i in range(train['n_iter']):
            gibbs_step()
            sample['assignments'][i, :] = state['assignments']

            print('train | iter {}, number of clusters {}, cluster indices {}\n{}'.format(i, state['n_clusters'], state['cluster_ids'], state['assignments'], ))

            # check if number of clusters is correct
            try: state['n_clusters'] == len(set(state['assignments']))
            except:
                print('wrong number of clusters')
                print(state['n_clusters'])
                print(len(set(state['assignments'])))
                exit(1)

        #     from collections import Counter
        #     if i % 10 == 0:
        #         plt.clf()
        #         argmax = [Counter(sample['assignments'][:i + 1, j].tolist()).most_common(1)[0][0] for j in range(n_data)]
        #         plt.scatter(data, y=[0] * len(data), c=argmax, s=10, cmap='gist_rainbow')
        #         plt.pause(0.05)
        #
        # plt.show()

    gibbs()

    return state


if __name__ == '__main__':
    edges, adj = get_data('sbm')
    state =mdnd(10, np.array(edges))

    import networkx as nx
    g = nx.from_numpy_matrix(np.matrix(adj))
    # nx.draw(g)
    # plt.show()
    pos = nx.spring_layout(g)
    # nx.draw(g)
    from collections import Counter
    mode = Counter(state['cluster_ids']).most_common()[0][0]
    print(mode)
    a = edges[state['assignments'] == mode]
    a = [tuple(b) for b in a]
    print(a)

    nx.draw_networkx_edges(g, pos, edgelist=a, edge_color='r', alpha=0.5)
    plt.show()

