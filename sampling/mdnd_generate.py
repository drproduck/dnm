from sampling.Process import *
import matplotlib.pyplot as plt


# Mixture od Dirichlet network
# print out 2 images of the random adjacency matrix and graph egdes, colored according to their cluster indicator c

# play with these parameters:
gamma = 1000 # control number of nodes
alpha = 1 # control number of clusters
tau = 5 # control cluster overlap

inlinks = dict()
outlinks = dict()
H = DirichletProcessDiscrete(ap=gamma) #control number of nodes
D = DirichletProcessDiscrete(ap=alpha)# control number of clusters

n = 200 # number of edges
edges = np.zeros((n,2), dtype=int)

# cluster indicators are sorted so that adjacency matrix has blocks
cs = sorted([D.sample() for _ in range(n)])
print(cs)
for i,c in enumerate(cs):
    if c not in inlinks.keys():
        inlinks[c] = DirichletProcess(H,ap=tau) # contrl inlink cluster overlap
    u = inlinks[c].sample()

    if c not in outlinks.keys():
        outlinks[c] = DirichletProcess(H,ap=tau) # control outlink cluster voerlap
    v = outlinks[c].sample()

    edges[i,:] = [u,v]

sz0 = max(edges[:,0])
sz1 = max(edges[:,1])
sz = max(sz0, sz1) + 1
adj = np.zeros((sz, sz), dtype=int)
mc = max(cs) + 1 # max of cs

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
# G = nx.from_edgelist(edges)
G = nx.MultiDiGraph(adj)
# print(len(edges))
# print(len(G.edges), G.edges)
# print(len(cs),cs)
edge_tuple = []
for i in range(len(edges)):
    edge_tuple.append(tuple(edges[i,:]))
print(edge_tuple)

pos = nx.spring_layout(G)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_nodes(G, pos, node_size=10)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
# print(edges)
# f = open('mdnd','w')
# for e in edges:
#     f.write('{} {}\n'.format(e[0], e[1]))
# # get the adj matrix
# plt.figure(2)
# plt.imshow(adj)
plt.show()
