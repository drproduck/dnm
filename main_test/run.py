import numpy as np
import matplotlib.pyplot as plt
from trunfreeDirichletMix import trunfreeInfiniteClusterDirichletMix
import math
from matplotlib import colors
from collections import Counter
import networkx as nx

def get_data_will(fname, ratio):
    """get data edge list from file. should return an (n,2) array of edges"""
    f = open(fname, 'r')
    [next(f) for _ in range(5)]
    n_edges = int(f.readline())
    print(n_edges)

    res = []
    for _ in range(n_edges):
        line = list(map(int, f.readline().strip().split(' ')))
        res += [line]
    res = np.array(res)
    links = res[:,:2]
    clusters = res[:,2]
    nodes = list(set(links.flatten()))
    sz = int(links.flatten().max() + 1)
    adj = np.zeros((sz, sz), dtype=int)

    # node clusters
    next(f) # skip blank line
    node_cluster = [list(map(int, line.strip().split(' '))) for line in f]

    idxperm = np.random.permutation(len(links))
    cutp = math.floor(len(links) * ratio)
    idx_train = idxperm[:cutp]
    idx_test = idxperm[cutp:]
    links_train = links[idx_train,:]
    links_test = links[idx_test,:]
    clusters_train = clusters[idx_train]
    clusters_test = clusters[idx_test]

    for e,c in zip(links,clusters):
        adj[e[0], e[1]] = c + 5

    return links_train,links_test,clusters_train,clusters_test,nodes, node_cluster


def get_data_simple(fname):
    """get data edge list from file. should return an (n,2) array of edges"""
    f = open(fname, 'r')
    res = np.array([list(map(int, line.strip().split(' '))) for line in f], dtype=int)
    nodes = list(set(res.flatten()))
    sz = int(res.flatten().max() + 1)
    adj = np.zeros((sz, sz), dtype=int)

    for e in res:
        adj[e[0], e[1]] += 1

    return res, nodes, adj

def display_adjacency(edges, clusters=None):

    cmap = colors.ListedColormap(['white', 'red', 'green', 'blue','yellow','purple','orange','brown','black'])
    bounds = [0,1,2,3,4,5,6,7,8,9]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    count = Counter(clusters).most_common()
    count = [c[0] for c in count]
    temp = [1,2,3,4,5,6,7,8,9]
    color = {count[x]: temp[x] if x < 8 else temp[8] for x in range(len(count))}
    sz = edges.max(axis=1).max() + 1
    adj = np.zeros([sz,sz],dtype=int)
    for e,c in zip(edges, clusters):
        adj[e[0], e[1]] = color[c]
    plt.imshow(adj,cmap=cmap,norm=norm)
    plt.show()

def display_graphx(edges, clusters=None):
    plt.set_cmap('gist_rainbow')
    g = nx.DiGraph()
    unique_edges = []
    unique_clusters = []
    for i,e in enumerate(edges):
        if g.has_edge(e[0], e[1]):
            continue
        else:
            g.add_edge(e[0], e[1])
            unique_edges += [(e[0], e[1])]
            unique_clusters += [clusters[i]]

    pos = nx.spring_layout(g)
    node = sorted([i for i in pos.keys()])
    nx.draw_networkx_nodes(g, pos, nodelist=node, node_size=30)
    nx.draw_networkx_edges(g, pos, edgelist=unique_edges, edge_color=unique_clusters, width=0.5)
    plt.show()

if __name__ == '__main__':
    links_train,links_test,clusters_train,clusters_test,nodes,node_cluster = get_data_will('toy_test', ratio=1)
    # links_train = links_train.tolist()
    # trunfreeInfiniteClusterDirichletMix()
    print(node_cluster)
    display_graphx(links_train, clusters_train)


