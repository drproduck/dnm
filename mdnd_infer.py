import numpy as np
from numpy.random import choice, dirichlet
import matplotlib.pyplot as plt
import networkx as nx
from Process import Antoniak



# This is the main inference algorith mfor mixture of dirichlet networks

def get_data(fname):
    """get data edge list from file. should return an (n,2) array of edges"""
    f = open(fname, 'r')
    res = np.array([list(map(int, line.strip().split(' '))) for line in f], dtype=int)
    sz = int(res.flatten().max() + 1)
    adj = np.zeros((sz, sz), dtype=int)

    for e in res:
        adj[e[0], e[1]] += 1

    return res, adj


def mdnd(n_clusters, edges):
    """main algorithm"""
    n_edges = edges.shape[0]

    node_ids = list(set(edges.flatten())) + [-1] # funny stuff: node is like topic in lda, while lda topics can grow in sampling time nodes dont
    n_nodes = len(node_ids) - 1
    antoniak = Antoniak() # antoniak distribution
    dirich = dirichlet(alpha=np.ones(n_nodes + 1))
    beta = {node_ids[i]: dirich[i] for i in range(n_nodes + 1)}

    # store variable that will be sampled
    state = {
        'cluster_ids': [i for i in range(n_clusters)],
        'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
        # the cluster assignment of each edge
        'n_clusters': n_clusters,
        'sizes': {s: 0 for s in range(n_clusters)},  # size of each cluster
        'beta': beta
    }

    # store hyperparameters (unchanged)
    fixed = {
        'alpha_D': 10, # control the number of clusters
        'tau': 10, # control cluster overlap
        'gamma_H': 10, # control number of nodes
        'edges': edges,
        'n_edges': n_edges,
        # 'node_indices': set(),
        'node_weights': dict(),

        # update these, else very slow
        'inlinks': dict(),
        'outlinks': dict()
    }

    # train parameters
    train = {
        'n_iter': 20,
        'n_burnin': 0,
    }

    # samples from samplers i.e number of samples = number of sampling iterations
    sample = {
        'assignments': np.zeros((train['n_iter'], n_edges), dtype=int),
    }

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
        if ind == -1: return sum(res)
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
        if ind == -1: return sum(res)
        res[ind] = False
        return sum(res)

    def sample_outlink_size(u):
        """
        step 7 of update_network_model
        compute \sum_k ro_i.^k
        """
        ro_sum_k = 0
        for c in state['cluster_ids']:
            Lu = get_outlink_size(u, c, -1)
            # if u == 19:
            #     print('19: outlink size {} ,cluster {}'.format(Lu, c))
            #     print(state['assignments'])
            if Lu > 1:
                alpha = fixed['tau'] * state['beta'][u]
                ro = antoniak.sample(alpha=alpha, n=Lu)
                ro_sum_k += ro
            else:
                ro_sum_k += Lu
        return ro_sum_k

    def sample_inlink_size(v):
        """
        step 7 of update_network_model
        compute \sum_k ro_.i^k
        """
        ro_sum_k = 0
        for c in state['cluster_ids']:
            Lv = get_outlink_size(v, c, -1)
            # if v == 19:
            #     print('19: inlink size {}, cluster {}'.format(Lv, c))
            #     print(state['assignments'])
            if Lv > 1:
                alpha = fixed['tau'] * state['beta'][v]
                ro = antoniak.sample(alpha=alpha, n=Lv)
                ro_sum_k += ro
            else:
                ro_sum_k += Lv
        return ro_sum_k

    def sample_node_prob():
        """step 8 of update_network_model, sample new node probability beta"""
        ro = []
        for node in node_ids:
            if node == -1: continue
            ro.append(sample_outlink_size(node) + sample_inlink_size(node))

        # print(ro)
        ro.append(fixed['gamma_H'])
        beta = dirichlet(alpha=ro)

        for i,node in enumerate(node_ids):
            state['beta'][node] = beta[i]


    def get_assignment_prob(cluster, ind):
        """step 5 of update_network_model"""
        u = fixed['edges'][ind][0]
        v = fixed['edges'][ind][1]

        # new cluster
        if cluster == -1:
            return np.log(fixed['alpha_D']) + 2 * np.log(fixed['tau']) + np.log(state['beta'][u]) + np.log(state['beta'][v])

        else:
            if state['assignments'][ind] == cluster:
                Nk = state['sizes'][cluster] - 1
            else:
                Nk = state['sizes'][cluster]

            Lu = get_outlink_size(u, cluster, ind)
            Lv = get_inlink_size(v, cluster, ind)

            return np.log(Nk) + np.log(Lu + fixed['tau'] * state['beta'][u]) + np.log(Lv + fixed['tau'] * state['beta'][v])

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

    def sample_assignment(ind):
        """
        step 5
        sample new cluster assignment
        """

        old_assignment = state['assignments'][ind]
        assignment_probs = [get_assignment_prob(c, ind) for c in (state['cluster_ids'] + [-1])]

        # to reduce underflow
        assignment_probs = assignment_probs - max(assignment_probs)
        assignment_probs = np.exp(assignment_probs)

        new_assignment = choice(state['cluster_ids'] + [-1], p=assignment_probs / sum(assignment_probs))

        re_cluster(new_assignment, old_assignment, ind)


    def gibbs_step():
        """1 step of Gibbs, sample for each edge randomly."""

        for ind in np.random.permutation(n_edges):
            sample_assignment(ind) # sample c
            sample_node_prob() # sample beta

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
            print('beta',state['beta'])

            # check if number of clusters is correct
            try: state['n_clusters'] == len(set(state['assignments']))
            except:
                print('wrong number of clusters')
                print(state['n_clusters'])
                print(len(set(state['assignments'])))
                exit(1)

            from collections import Counter
            plt.clf()
            adj = np.zeros((n_nodes, n_nodes), dtype=int)
            for e,c in zip(edges, state['assignments']):
                adj[e[0], e[1]] = c
            plt.imshow(adj)
            plt.pause(1)

        plt.show()

    gibbs()

    return state


if __name__ == '__main__':
    edges, adj = get_data('sbm')
    state =mdnd(len(edges), np.array(edges))

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

