"""make artificial data using stochastic block model"""
from numpy.random import *
import matplotlib.pyplot as plt
import numpy as np

K = 5
M = 30

phi = np.random.randint(1,10,size=K)
alpha = 1
beta = 100

# cluster parameter
theta = np.random.beta(alpha,beta,(K,K))
for i in range(K):
    theta[i,i] += (1 - theta[i,i]) / 2

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

# adj = np.maximum(adj, adj.T)
f.close()
plt.spy(adj)
plt.show()