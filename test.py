import matplotlib.pyplot as plt
from numpy.random import *
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from Process import *
import matplotlib.pyplot as plt
# stick = StickBreakingProcess(alpha=100)
# sample = stick.sample(1000)
# plt.hist(sample, bins=100)
# plt.show()

# g = Gaussian(1,2)
# hdp = HierarchicalDirichletProcess(10,10,g)
# plt.hist([hdp.sample() for i in range(1000)],bins=100)
# plt.show()

# Mixture od Dirichlet network

inlinks = dict()
outlinks = dict()
H = DirichletProcessDiscrete(ap=100)
D = DirichletProcessDiscrete(ap=10)

n = 1000 # number of edges
edges = np.zeros((n,2), dtype=int)
cs = sorted([D.sample() for _ in range(n)])
print(cs)

for i,c in enumerate(cs):
    if c not in inlinks.keys():
        inlinks[c] = DirichletProcess(H,ap=5)
    u = inlinks[c].sample()

    if c not in outlinks.keys():
        outlinks[c] = DirichletProcess(H,ap=5)
    v = outlinks[c].sample()

    edges[i,:] = [u,v]

sz0 = max(edges[:,0])
sz1 = max(edges[:,1])
print(sz0, sz1)

print(edges)
adj = np.zeros((sz0+1,sz1+1), dtype=int)
for i in range(n):
    adj[edges[i,0], edges[i,1]] += 1
print(adj)
# plt.imshow(adj,cmap='gist_rainbow')
plt.spy(adj)
plt.show()


# a = DirichletProcessDiscrete(ap=10)
# plt.hist([a.sample() for _ in range(1000)], bins=100)
# plt.show()
