"""
distance dependent mixture of Dirichlet network distributions
"""
from collections import Counter
from scipy.misc import logsumexp
from ddcrp_mdnd.my_util import *
import networkx as nx
import numpy as np


class Model:

    def __init__(self, X_train, distance, decay_fn, params, init=None):
        """
        maximum number of clusters = number of edges (happens when all are self-links)
        :param X_train: 2d array of edges
        :param distance: distance function
        :param decay_fn: decay function
        :param params: params object containing: alpha, gamma and tau.
        :param init: initial number of clusters
        """
        self.params = params

        self.num_vertices = np.max(X_train) + 1
        self.num_edges = len(X_train)
        self.source_vertices = X_train[:, 0]
        self.sink_vertices = X_train[:, 1]
        self.tables = np.zeros(self.num_vertices, dtype=int)

        # tables are used when sampling new beta. tables don't change
        for edge in range(self.num_edges):
            self.tables[self.source_vertices[edge]] += 1
            self.tables[self.sink_vertices[edge]] += 1
        self.tables = self.tables.tolist()

        # aux variable
        self.beta = np.ones(self.num_vertices + 1)
        self.beta = self.beta / self.beta.sum()

        # initialize prior matrix
        self.prior = np.zeros([self.num_edges, self.num_edges], dtype=float)
        for i in range(self.num_edges):
            for j in range(self.num_edges):
                if i == j:
                    self.prior[i, j] = np.log(params.alpha)
                else:
                    self.prior[i, j] = np.log(decay_fn(distance(i, j)))

        # initialize cluster indicators for edges, links, source_count ,sink_count, source and sink likelihood and
        # cluster likelihood, unique_clusters
        try:
            if isinstance(init, np.ndarray):
                """This initialization assumes links are given as init"""
                self.cluster_of_edge = np.zeros(self.num_edges, dtype=int)
                self.link_of_edge = init
                i = 0
                self.graph = nx.from_edgelist([(a, b) for a, b in zip(np.arange(len(init), dtype=int), init)], nx.MultiGraph)
                for component in nx.connected_components(self.graph):
                    self.cluster_of_edge[list(component)] = i
                    i += 1
                self.unique_clusters = set(np.arange(i, dtype=int))
                print('clusters:', self.cluster_of_edge)

                # # BFS
                # i = 0
                # self.cluster_of_edge = np.zeros(self.num_edges, dtype=int)
                # visited = np.zeros(self.num_edges, dtype=bool)
                # for e in range(self.num_edges):
                #     if not visited[e]:
                #         i += 1
                #         linked = [e]
                #         while len(linked) != 0:
                #             l = linked.pop()
                #             if not visited[init[l]]:
                #                 visited[init[l]] = True
                #                 self.cluster_of_edge[init[l]] = i
                #                 linked += [init[l]]
                # self.unique_clusters = set(np.arange(i, dtype=int))
                # print(self.cluster_of_edge)

            else:
                """This initialization has only K clusters"""
                stored = dict()
                self.cluster_of_edge = np.random.choice(init, size=self.num_edges)
                self.link_of_edge = np.arange(self.num_edges, dtype=int)
                for i in range(init):
                    linked = np.where(self.cluster_of_edge == i)[0]

                    if len(linked) == 1:
                        self.link_of_edge[linked[0]] = linked[0]  # self-link
                    else:
                        for li in linked:
                            self.link_of_edge[li] = np.random.choice(linked, 1)
                            while self.link_of_edge[li] == li:
                                self.link_of_edge[li] = np.random.choice(linked, 1)
                    stored[linked.min()] = linked

                for k, linked in stored.items():
                    self.cluster_of_edge[linked] = k

                self.unique_clusters = set(stored.keys())
                # print('cluster ', self.cluster_of_edge)

        except:
            """for initialization, each customer self-links"""
            self.link_of_edge = np.arange(self.num_edges, dtype=int)
            self.cluster_of_edge = np.arange(self.num_edges, dtype=int)
            self.unique_clusters = set(np.arange(self.num_edges, dtype=int))
            self.graph = nx.from_edgelist([(a, b) for a, b in zip(np.arange(len(self.link_of_edge), dtype=int), self.link_of_edge)], nx.MultiGraph)

        # finish initialization

        self.source_llhood = np.zeros(self.num_edges, dtype=float)
        self.sink_llhood = np.zeros(self.num_edges, dtype=float)
        self.cluster_llhood = np.zeros(self.num_edges, dtype=float)
        self.compute_llhood()

    def compute_llhood(self):
        # self.source_count = np.zeros((self.num_edges, self.num_vertices), dtype=int)
        # self.sink_count = np.zeros((self.num_edges, self.num_vertices), dtype=int)
        # for edge in range(self.num_edges):
        #     self.source_count[self.cluster_of_edge[edge]][self.source_vertices[edge]] += 1
        #     self.sink_count[self.cluster_of_edge[edge]][self.sink_vertices[edge]] += 1

        for cluster in self.unique_clusters:
            self.source_llhood[cluster] = self.hdp_llhood_vector(self.source_vertices[self.cluster_of_edge == cluster])
            self.sink_llhood[cluster] = self.hdp_llhood_vector(self.sink_vertices[self.cluster_of_edge == cluster])
            self.cluster_llhood[cluster] = self.source_llhood[cluster] + self.sink_llhood[cluster]

            # self.source_llhood[cluster] = self.hdp_llhood_count(self.source_count[cluster])
            # self.sink_llhood[cluster] = self.hdp_llhood_count(self.sink_count[cluster])

    def hdp_llhood_vector(self, obs):
        """
        compute log likelihood of the hierarchical DP
        :param obs: a vector of vertices
        :return:
        """
        count = Counter()
        lhood = 0
        for i in range(len(obs)):
            lhood += np.log(count[obs[i]] + self.params.tau * self.beta[obs[i]]) - np.log(i + self.params.tau)
            count[obs[i]] += 1
        return lhood

    def hdp_llhood_count(self, counts):
        """
        compute log likelihood of the hierarchical DP
        :param counts: a vector of counts
        :return:
        """
        lhood = 0
        total_count = 0
        for i, count in enumerate(counts):
            for c in range(count):
                lhood += np.log(c + self.params.tau * self.beta[i]) - np.log(total_count + self.params.tau)
                total_count += 1
        return lhood

    def remove_link(self, edge, params):
        """
        :param edge: index of the edge which link is re-sampled
        :param params:
        :return:
        """
        # remove this edge's link by (temporarily) setting its link and cluster to itself
        old_link = self.link_of_edge[edge]
        self.link_of_edge[edge] = edge
        self.unique_clusters -= {self.cluster_of_edge[edge]}
        self.graph.remove_edge(edge, old_link)
        self.graph.add_edge(edge, edge)
        linked_edges = connected_component(edge, self.graph)  # linked edges after removing link
        self.cluster_of_edge[linked_edges] = edge
        self.unique_clusters |= {edge}

        if old_link not in linked_edges:
            # iff there is a table split
            unlinked_edges = connected_component(old_link, self.graph)
            self.cluster_of_edge[unlinked_edges] = old_link  # avoid when their clusters all pointed to this edge
            self.unique_clusters |= {old_link}

            self.source_llhood[old_link] = self.hdp_llhood_vector(self.source_vertices[unlinked_edges])
            self.sink_llhood[old_link] = self.hdp_llhood_vector(self.sink_vertices[unlinked_edges])
            self.cluster_llhood[old_link] = self.source_llhood[old_link] + self.sink_llhood[old_link]

            self.source_llhood[edge] = self.hdp_llhood_vector(self.source_vertices[linked_edges])
            self.sink_llhood[edge] = self.hdp_llhood_vector(self.sink_vertices[linked_edges])
            self.cluster_llhood[edge] = self.source_llhood[edge] + self.sink_llhood[edge]

            # # update count
            # self.update_count(linked_edges, edge)
            # self.update_count(unlinked_edges, old_link)

        return linked_edges

    def update_count(self, edges, cluster):
        """
        update source and sink count

        :param edges: edges in this cluster
        :param cluster:
        :return:
        """
        so_count = Counter(self.source_vertices[edges])
        self.source_count[cluster] = np.zeros(self.num_vertices, dtype=int)
        for k, v in so_count.items():
            self.source_count[cluster][k] = v
        self.sink_count[cluster] = np.zeros(self.num_vertices, dtype=int)
        si_count = Counter(self.sink_vertices[edges])
        for k, v in si_count.items():
            self.sink_count[cluster][k] = v

    def sample_link_collapsed(self, edge, params, linked_edges, test_chosen=None):
        """
        sample this edge's link (after it was removed)
        :param edge:
        :param params:
        :param linked_edges: edges connected  with this edge before sampling
        :param test_chosen: force connection for testing
        :return:
        """
        linked_source_vertices = self.source_vertices[linked_edges]
        linked_sink_vertices = self.sink_vertices[linked_edges]
        merged_source_llhood = np.zeros(self.num_edges, dtype=float)
        merged_sink_llhood = np.zeros(self.num_edges, dtype=float)
        merged_llhood = np.zeros(self.num_edges, dtype=float)

        for other_cluster in self.unique_clusters:
            if other_cluster != self.cluster_of_edge[edge]:
                newly_linked_source_vertices = self.source_vertices[self.cluster_of_edge == other_cluster]
                merged_source_vertices = np.concatenate((newly_linked_source_vertices, linked_source_vertices), axis=0)
                merged_source_llhood[other_cluster] = self.hdp_llhood_vector(merged_source_vertices)

                newly_linked_sink_vertices = self.sink_vertices[self.cluster_of_edge == other_cluster]
                merged_sink_vertices = np.concatenate((newly_linked_sink_vertices, linked_sink_vertices), axis=0)
                merged_sink_llhood[other_cluster] = self.hdp_llhood_vector(merged_sink_vertices)

                merged_llhood[other_cluster] = merged_source_llhood[other_cluster] + merged_sink_llhood[other_cluster]

        log_prob = np.zeros(self.num_edges, dtype=float)
        for other_edge in range(self.num_edges):
            this_cluster = self.cluster_of_edge[edge]
            other_cluster = self.cluster_of_edge[other_edge]
            if other_cluster == this_cluster:
                log_prob[other_edge] = self.prior[edge][other_edge]
            else:
                log_prob[other_edge] = self.prior[edge][other_edge] + merged_llhood[other_cluster] \
                                       - self.cluster_llhood[this_cluster] \
                                       - self.cluster_llhood[other_cluster]
                # print('connected to {}, prior {}, merged hood {}, this cluster {}, other cluster {}'.format(other_edge, self.prior[edge][other_edge], merged_llhood[other_cluster],
                #                                                                                   self.cluster_llhood[this_cluster],self.cluster_llhood[other_cluster]))

        # print(log_prob)
        log_prob -= logsumexp(log_prob)
        # test
        if test_chosen:
            self.link_of_edge[edge] = test_chosen
            self.graph.remove_edge(edge, edge)
            self.graph.add_edge(edge, test_chosen)
        else:
            self.link_of_edge[edge] = np.random.choice(self.num_edges, size=1, p=np.exp(log_prob))
            self.graph.remove_edge(edge, edge)
            self.graph.add_edge(edge, self.link_of_edge[edge])
            # self.link_of_edge[edge] = np.searchsorted(np.cumsum(np.exp(log_prob)), np.random.random())

        # update if the new link merge 2 clusters
        new_cluster = self.cluster_of_edge[self.link_of_edge[edge]]

        if new_cluster != edge:
            self.unique_clusters -= {edge}
            self.unique_clusters |= {new_cluster}
            self.cluster_of_edge[linked_edges] = new_cluster
            self.source_llhood[new_cluster] = merged_source_llhood[new_cluster]
            self.sink_llhood[new_cluster] = merged_sink_llhood[new_cluster]
            self.cluster_llhood[new_cluster] = merged_llhood[new_cluster]

            # # update count
            # self.update_count(np.where(self.cluster_of_edge == new_cluster)[0], new_cluster)

    def sample_beta(self):
        """the simple version, accumulate tables and sample"""
        self.beta = np.random.dirichlet(self.tables + [self.params.gamma])



    # def sample_beta(self, params):
    #     tables = np.zeros(self.num_vertices + 1)
    #     for cluster in self.unique_clusters:
    #         for vertex in range(self.num_vertices):
    #             if self.source_count[cluster][vertex] == 1:
    #                 tables[vertex] += 1
    #             elif self.source_count[cluster][vertex] > 1:
    #                 tmp = self.partitionCRP(params.alpha, self.source_count[cluster][vertex])
    #                 tables[vertex] += tmp
    #             if self.sink_count[cluster][vertex] == 1:
    #                 tables[vertex] += 1
    #             elif self.sink_count[cluster][vertex] > 1:
    #                 tmp = self.partitionCRP(params.alpha, self.sink_count[cluster][vertex])
    #                 tables[vertex] += tmp
    #
    #     tables[self.num_vertices] = params.gamma
    #     self.beta = np.random.dirichlet(tables)
    #     if any(self.beta == 0):
    #         raise Warning('something might be wrong')

    # @staticmethod
    # def partitionCRP(alpha, N):
    #     K = 0
    #
    #     for iter in range(int(N)):
    #         r = np.random.rand()
    #         p = alpha / (alpha + iter)
    #         if r < p:
    #             K += 1
    #
    #     return K

# class Summary():
