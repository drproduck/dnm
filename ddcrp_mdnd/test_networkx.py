import networkx as nx
import numpy as np

def from_np_to_tuple(edge_arr):
    res = []
    for x,y in  edge_arr:
        res += [(x,y)]
    return res

# X_train = np.array([[0, 0, 1, 2, 3, 2], [4, 3, 0, 0, 1, 2]]).reshape((6,2))
# X_train = from_np_to_tuple(X_train)

init = np.array([2,0,1,3,5,3])
edgelist = [(0,2),(1,1),(2,1),(3,5),(4,5),(5,3)]
G = nx.from_edgelist(edgelist, nx.MultiGraph)
print(nx.number_connected_components(G))
print(list(nx.connected_components(G)))
print(nx.node_connected_component(G, 4))
print(G.edges)
