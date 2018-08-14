"""make artificial data using stochastic block model"""
from numpy.random import *
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
K = 7
M = 100

# phi = np.random.randint(1,10,size=K)
phi = np.array([10]*K,dtype=float)
alpha = 1
beta = 10

# cluster parameter
theta = np.random.beta(alpha,beta,(K,K))
# for i in range(K):
#     theta[i,i] += (1 - theta[i,i]) /

pi = dirichlet(phi)

# cluster proportion
c = choice(K,p=pi, replace=True, size=M)
c = sorted(c)
print(c)

adj = np.zeros((M,M))
for i in range(M):
    for j in range(M):
        tt = theta[c[i],c[j]]
        adj[i,j] = choice(2, p=[1-tt, tt])

f = open('sbm', 'w')
for i in range(M):
    for j in range(M):
        if adj[i,j] == 1:
            f.write('{} {}\n'.format(i,j))

# graph = nx.MultiDiGraph(adj)
# pos = nx.spring_layout(graph)
# nx.draw_networkx_nodes(graph, pos, node_size=10)
# nx.draw_networkx_edges(graph, pos)
# plt.show()
adj = np.maximum(adj, adj.T)
f.close()
plt.spy(adj)
plt.show()