import matplotlib.pyplot as plt
from numpy.random import *
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from Process import *
import matplotlib.pyplot as plt


# Mixture od Dirichlet network
# print out 2 images of the random adjacency matrix and graph egdes, colored according to their cluster indicator c

inlinks = dict()
outlinks = dict()
H = DirichletProcessDiscrete(ap=1000) #control number of nodes
D = DirichletProcessDiscrete(ap=1)# control number of clusters

n = 500 # number of edges
edges = np.zeros((n,2), dtype=int)

# cluster indicators are sorted so that adjacency matrix has blocks
cs = sorted([D.sample() for _ in range(n)])
print(cs)

for i,c in enumerate(cs):
    if c not in inlinks.keys():
        inlinks[c] = DirichletProcess(H,ap=5) # contrl inlink cluster overlap
    u = inlinks[c].sample()

    if c not in outlinks.keys():
        outlinks[c] = DirichletProcess(H,ap=5) # control outlink cluster voerlap
    v = outlinks[c].sample()

    edges[i,:] = [u,v]

sz0 = max(edges[:,0])
sz1 = max(edges[:,1])
sz = max(sz0, sz1) + 1
adj = np.zeros((sz, sz), dtype=int)
mc = max(cs) # max of cs

i = 0
while i < len(edges):
    if adj[edges[i,0], edges[i,1]] > 0:
        del cs[i]
        edges = np.delete(edges, i, 0)
    else:
        adj[edges[i,0], edges[i,1]] = mc - cs[i]
        i+=1

import networkx as nx
plt.figure(1)
adj = np.matrix(adj)
G = nx.from_edgelist(edges)

# print(len(edges))
# print(len(G.edges), G.edges)
# print(len(cs),cs)
edge_tuple = []
for i in range(len(edges)):
    edge_tuple.append(tuple(edges[i,:]))
print(edge_tuple)

pos = nx.spring_layout(G, 2)
nx.draw_networkx_edges(G, pos, edgelist=edge_tuple, edge_color=cs)
nx.draw_networkx_nodes(G, pos, node_size=1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
print(edges)

# get the adj matrix
plt.figure(2)
plt.imshow(adj)
plt.show()
