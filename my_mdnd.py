from matplotlib import colors
import numpy as np
from numpy.random import choice, dirichlet
import matplotlib.pyplot as plt
import networkx as nx
from Process import Antoniak
from collections import Counter


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

    # set up some layout for graph plot
    edge_tuple = []
    for i in range(len(edges)):
        edge_tuple.append(tuple(edges[i, :]))

    G = nx.from_edgelist(edges)
    pos = nx.spring_layout(G, 2)
    pos[0] = np.array([0,0])
    pos[2] = np.array([1,0])
    pos[1] = np.array([0.5,1])
    pos[6] = np.array([0.5,2])
    pos[7] = np.array([0.5,3])
    pos[8] = np.array([0.5,4])
    pos[3] = np.array([0.5,5])
    pos[4] = np.array([0,6])
    pos[5] = np.array([1,6])


    # store variable that will be sampled
    state = {
        'cluster_ids': [i for i in range(n_clusters)],
        'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_edges)], dtype=int),
        # the cluster assignment of each edge
        'n_clusters': n_clusters,
        'sizes': {s: 0 for s in range(n_clusters)},  # size of each cluster
        'beta': None # set later
    }

    # store hyperparameters (unchanged)
    fixed = {
        'alpha_D': 1, # control the number of clusters
        'tau': 1, # control cluster overlap
        'gamma_H': 1, # control number of nodes
        'edges': edges,
        'n_edges': n_edges,
        'node_ids': None,
        'node_cardinals': None,
        'node_weights': dict(),

        # update these, else very slow
        'inlinks': dict(),
        'outlinks': dict()
    }

    # train parameters
    train = {
        'n_iter': 1000,
        'n_burnin': 100,
    }

    # samples from samplers i.e number of samples = number of sampling iterations
    sample = {
        'assignments': np.zeros((train['n_iter'], n_edges), dtype=int),
        'beta': None
    }

    # collect node ids, number of node_id and save
    node_cardinals = Counter(list(edges.flatten()))
    n_nodes = len(node_cardinals) # note counting latent weight
    node_cardinals[-1] = fixed['gamma_H'] # this is the latent weight gamma
    fixed['node_cardinals'] = node_cardinals

    # 1 sample of beta and save
    dirich = dirichlet(alpha=[i for i in node_cardinals.values()])
    fixed['node_ids'] = [i for i in node_cardinals.keys()] # save node_ids here so that it includes -1 and aligns with node_cardinal_values
    state['beta'] = {fixed['node_ids'][i]: dirich[i] for i in range(n_nodes + 1)}
    sample['beta'] = np.zeros((train['n_iter'], n_nodes+1), dtype=np.float)


    def get_outlink_size_in_cluster(u, k):
        """:return: # size of outlink node u in cluster k"""

        res = [a and b for a, b in zip(fixed['edges'][:,0] == u, state['assignments'] == k)]

        return sum(res)

    def get_inlink_size_in_cluster(v, k):
        """:return: # size of inlink node v in cluster k"""

        res = [a and b for a, b in zip(fixed['edges'][:, 1] == v, state['assignments'] == k)]

        return sum(res)


    def sample_node_prob():
        """sample new node probability beta, THIS IS EXPERIMENTAL"""

        dirich = dirichlet(alpha=[i for i in node_cardinals.values()])
        state['beta'] = {fixed['node_ids'][i]: dirich[i] for i in range(n_nodes + 1)}
        return state['beta']

    def get_assignment_prob(cluster, ind):
        """sample from assignment conditional"""

        u = fixed['edges'][ind][0]
        v = fixed['edges'][ind][1]

        # new cluster
        if cluster == -1:
            return np.log(fixed['alpha_D']) + np.log(state['beta'][u]) + np.log(state['beta'][v])

        else:

            # size of this cluster except this edge is necessary
            if state['assignments'][ind] == cluster:
                Nk = state['sizes'][cluster] - 1
            else:
                Nk = state['sizes'][cluster]

            # this one is tricky ...
            Lu = get_outlink_size_in_cluster(u, cluster)
            # if u has been seen in this cluster A_k before
            if Lu > 0:
                # discount
                if state['assignments'][ind] == cluster:
                    wu = (Lu - 1) / (Nk + fixed['tau'])
                    # wu = Lu - 1
                else:
                    wu = Lu / (Nk + fixed['tau'])
                    # wu = Lu
                # if Lu = 1 then better go to another cluster ...
            else:
                # if u has not been seen
                wu = fixed['tau'] / (Nk + fixed['tau']) * state['beta'][u]
                # wu = fixed['tau'] * state['beta'][u]

            # similarly for v
            Lv = get_inlink_size_in_cluster(v, cluster)
            # if v has been seen in this cluster B_k before
            if Lv > 0:
                # discount
                if state['assignments'][ind] == cluster:
                    wv = (Lv - 1) / (Nk + fixed['tau'])
                    # wv = Lv - 1
                else:
                    wv = Lv / (Nk + fixed['tau'])
                    # wv = Lv
                # if Lv = 1 then better go to another cluster ...
            else:
                # if v has not been seen
                wv = fixed['tau'] / (Nk + fixed['tau']) * state['beta'][v]
                # wv = fixed['tau'] * state['beta'][v]

            return np.log(Nk) + np.log(wu) + np.log(wv)

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
            sample['beta'][i, :] = [i for i in state['beta'].values()]

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
            # cm = plt.get_cmap('gist_rainbow')
            # cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
            # scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
            # color map
            # cmap = colors.ListedColormap(['white', 'red', 'green', 'blue','yellow'])
            # bounds = [0,1,2,3,4]
            # norm = colors.BoundaryNorm(bounds, cmap.N)
            #
            # nx.draw_networkx_nodes(G, pos, node_size=1)
            # nx.draw_networkx_edges(G, pos, edgelist=edge_tuple, edge_color=state['assignments'])
            # plt.xlim(0, 1)
            # plt.ylim(-1, 7)
            # plt.subplot(122)
            # adj = np.zeros((n_nodes, n_nodes), dtype=int)

            # color the 3 biggest clusters + 1 color for the rest
            # count = Counter(state['assignments'])
            # count = [i for i in count.keys()]
            # count = sorted(set(state['assignments']))
            # print(count)
            # temp = [1,2,3,4]
            # color = {count[x]: temp[x] if x < 3 else temp[3] for x in range(len(count))}
            # for e,c in zip(edges, state['assignments']):
            #     adj[e[0], e[1]] = color[c]
            # plt.imshow(adj, cmap=cmap, norm=norm)

            # for n,e in enumerate(edges):
            #     adj[e[0],e[1]] = state['assignments'][n] + 5
            #
            # plt.imshow(adj)
            #
            # plt.pause(1)

        # plt.show()

    gibbs()

    return state,sample,train


if __name__ == '__main__':
    edges, adj = get_data('bell')
    n_edges = len(edges)
    state,sample,train = mdnd(2, np.array(edges))
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


