#! /usr/local/bin/ipython
from os.path import isfile
import re
import itertools
#from urllib import urlopen
from math import log, ceil, lgamma
import random as rd
import numpy as np
import pdb
from scipy.special import gammaln, betaln
from scipy.misc import logsumexp
from pylab import *
from copy import copy, deepcopy
#import networkx as nx
import csv 
import operator
import time
from dd.network_classes import *
from collections import Counter


def gen_data(num_clusters, cluster_size,  num_per_cluster, overlap=0, bg_prob = 0, same_size=True):
    #generates toy data with num_cluster clusters of size cluster_size+overlap, and overlap of overlap. bg_prob adds background noise.
    links =[]
    clusters = []
    if same_size == True:
        num_users = num_clusters*cluster_size+overlap
        for i in range(num_clusters):
            for n in range(num_per_cluster):
                clusters.append(i)
                if np.random.rand()<bg_prob:
                    sr = rd.sample(range(num_users),2)
                else:
                    sr = rd.sample(range(i*cluster_size, (i+1)*cluster_size+overlap),2)
                links.append(sr)
    else:
        num_users = np.sum(cluster_size)+overlap
        node_clusters = {}
        pastclusters=0
        for i in range(num_clusters):
            for n in range(num_per_cluster[i]):
                clusters.append(i)
                if np.random.rand()<bg_prob:
                    sr = rd.sample(range(0,num_users),2)
                else:
                    sr = rd.sample(range(pastclusters, pastclusters+cluster_size[i]+overlap),2)
                links.append(sr)
                if sr[0] in node_clusters:
                    node_clusters[sr[0]] += [i]
                else:
                    node_clusters[sr[0]] = [i]

                if sr[1] in node_clusters:
                    node_clusters[sr[1]] += [i]
                else:
                    node_clusters[sr[1]] = [i]

            pastclusters+=cluster_size[i]
    
    Z = np.zeros((num_users,num_users))
    for link in links:
        Z[link[0],link[1]]+=1
    Z = Z.astype(int)
    return links, clusters, Z, node_clusters


def gibbs(Ztrain,init_K, alpha_params,gamma_params,tau_params,sample_params=True, symmetric=False, collapsed=True, maxit=1000, Ztest=None):
    Ztrain = np.array(Ztrain)
    params = Params(alpha_params, gamma_params, tau_params, sample_params, Ztrain,  symmetric, collapsed)

    model = Model(Ztrain, init_K, params)

    for iter in range(maxit):
        #if iter%10 == 0:
            #print 'iter: '+repr(iter)+'\t K: '+repr(model.K)+'\t alpha: '+repr(params.alpha)+'\t gamma: '+repr(params.gamma)+'\t tau: '+repr(params.tau)
        #pdb.set_trace()

        #sample cluster assignments
        for n in range(params.num_links):
            model.sample_link(Ztrain[n,:], n, params)
            #pdb.set_trace()

        #TODO: skip this for now
        if collapsed is True:
            model.split_merge(Ztrain, params)

        #sample beta
        model.sample_beta(params)
        if collapsed is False:
            model.sample_pie(params)
            model.count_tables(params)
            
        #sample hypers
        if params.sample_params is True:
            params.sample_alpha(model.K)
            try:
                num_tables = model.num_tables
            except TypeError:
                pdb.set_trace()
            
            params.sample_tau(model.K, model.cluster_counts,num_tables)
            params.sample_gamma(num_tables)

    if Ztest is not None:
        if params.collapsed is True:
            log_lik, lp_links = model.pred_ll_collapsed(Ztest,params)

        else:
            log_lik, lp_links = model.pred_ll_uncollapsed(Ztest, params)
        #print 'log_lik: '+repr(log_lik)
        #print '\n'
        #print lp_links

    return model, params

def will_gen_data(params):

    if params['num_clusters'] is None:
        num_clusters = 6
    else:
        num_clusters = params['num_clusters']
    if params['cluster_size'] is None:
        cluster_size=[5,5,5,5,5,5]
    else:
        cluster_size=params['cluster_size']
    if params['num_per_cluster'] is None:
        num_per_cluster=[20,20,20,20,20,20]
    else:
        num_per_cluster = params['num_per_cluster']
    if params['overlap'] is None:
        overlap = 0
    else:
        overlap = params['overlap']
    if params['bg_prob'] is None:
        bg_prob = 0.2
    else:
        bg_prob = params['bg_prob']
    links, clusters, Z, node_clusters = gen_data(num_clusters=num_clusters, cluster_size=cluster_size,
                                                 num_per_cluster=num_per_cluster, overlap=overlap,bg_prob=bg_prob,same_size=False)
    # node clusters
    idx_node = []
    nc = []
    for k,v in node_clusters.items():
        idx_node += [k]
        nc += [Counter(v).most_common(1)[0][0]]
    perm = np.argsort(np.array(idx_node, dtype=int))
    print(perm)
    print(idx_node)
    nc = np.array(nc)[perm]
    idx_node = np.array(idx_node)[perm]

    links = np.array(links, dtype=int)
    clusters = np.array(clusters, dtype=int)
    s = np.max(links, axis=0)

    idx = np.argsort(clusters)
    links = links[idx,:]
    clusters = clusters[idx]
    adj = np.zeros(s+1, dtype=int)
    for link, cluster in zip(links, clusters):
        adj[link[0], link[1]] = cluster + 5

    f = open('toy_test','w')
    f.write('number_clusters {}\n'.format(num_clusters))
    f.write('cluster_size {}\n'.format(cluster_size))
    f.write('num_per_cluster {}\n'.format(num_per_cluster))
    f.write('overlap {}\n'.format(overlap))
    f.write('bg_prob {}\n'.format(bg_prob))
    f.write(str(len(links))+'\n')
    for l,c in zip(links,clusters):
        f.write('{} {} {}\n'.format(l[0],l[1],c))
    f.write('\n')
    for id, c in zip(idx_node, nc):
        f.write('{} {}\n'.format(id, c))
    f.close()


