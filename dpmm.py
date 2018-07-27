import numpy as np
from numpy.random import choice, dirichlet, normal
import scipy.stats as stats
import matplotlib.pyplot as plt

def DirichletMixtureModel(data, n_data, n_clusters, scale=0.1):
    """dirichlet Process GMM with fixed sigma"""

    state = {
        'cluster_ids': [i for i in range(n_clusters)],
        'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_data)], dtype=int),
        # 'means': np.random.randn(n_clusters) * 2,
        'sizes': {s: 0 for s in range(n_clusters)},
        'sums': {s: 0 for s in range(n_clusters)},
        'n_clusters': n_clusters
    }
    fixed = {
        'sigma_data': 1,
        'sigma_mean': 1,
        'mean_mean': 1,
        'data': data,
        'alpha_dp': 1
    }

    train = {
        'n_iter': 500,
        'n_burnin': 10,
    }

    sample = {
        'assignments': np.zeros((train['n_iter'], n_data), dtype=int),
        'means': np.zeros((train['n_iter'], n_clusters)),
        'pi': np.zeros((train['n_iter'], n_clusters))
    }

    def z_posterior_predictive(cluster, data_ind):
        if cluster == -1:
            return np.log(fixed['alpha_dp'])
        else:
            if state['assignments'][data_ind] == cluster:
                return np.log(state['sizes'][cluster] - 1)
            else:
                return np.log(state['sizes'][cluster])

    def x_posterior_predictive(cluster, data_ind):
        if cluster == -1:
            return stats.norm.logpdf(fixed['data'][data_ind], fixed['mean_mean'], np.sqrt(fixed['sigma_mean']+fixed['sigma_data']))
        else:
            if state['assignments'][data_ind] == cluster:
                Nk = state['sizes'][cluster] - 1
                sumk = state['sums'][cluster]
            else:
                Nk = state['sizes'][cluster]
                sumk = state['sums'][cluster]
            sigk = 1 / (Nk / fixed['sigma_data'] + 1 / fixed['sigma_mean'])
            muk = sigk * (fixed['mean_mean'] / fixed['sigma_mean'] + sumk / fixed['sigma_data'])
            sigk += fixed['sigma_data']

            return stats.norm.logpdf(fixed['data'][data_ind], muk, np.sqrt(sigk))

    def re_cluster(new_cluster, old_cluster, data_ind):
        if new_cluster == old_cluster:
            return

        if new_cluster == -1:
            state['n_clusters'] += 1
            # print('new cluster')
            #create new cluster id
            new_cluster = max(state['cluster_ids']) + 1
            state['cluster_ids'] += [new_cluster]
            state['assignments'][data_ind] = new_cluster

            # update size
            state['sizes'][new_cluster] = 1
            state['sizes'][old_cluster] -= 1
            # update sum
            state['sums'][new_cluster] = fixed['data'][data_ind]
            state['sums'][old_cluster] -= fixed['data'][data_ind]

        else:
            # print('update')
            # update
            state['assignments'][data_ind] = new_cluster

            # update size
            state['sizes'][new_cluster] += 1
            state['sizes'][old_cluster] -= 1

            # update sum
            state['sums'][new_cluster] += fixed['data'][data_ind]
            state['sums'][old_cluster] -= fixed['data'][data_ind]

        # check if cluster needs prune
        if state['sizes'][old_cluster] == 0:
            # print('old cluster removed')
            state['n_clusters'] -= 1

            state['sizes'].pop(old_cluster, None)
            state['sums'].pop(old_cluster, None)
            state['cluster_ids'].remove(old_cluster)

        #diagnoze
        # c = dict()
        # for i in state['assignments']:
        #     if i in c:
        #         c[i] += 1
        #     else:
        #         c[i] = 1
        # for key in c:
        #     if not c[key] == state['sizes'][key]:
        #         print('something wrong', old_cluster, new_cluster)
        #         print(state['cluster_ids'])
        #         print('sizes tally', state['sizes'])
        #         print('sizes by assignments', c)
        #         # print(state)
        #         exit(0)
        #
        # d = dict()
        # for i in range(n_data):
        #     if state['assignments'][i] in d:
        #         d[state['assignments'][i]] += fixed['data'][i]
        #     else:
        #         d[state['assignments'][i]] = fixed['data'][i]
        # for key in d:
        #     if not abs(d[key] - state['sums'][key]) < 0.00001:
        #         print('something wrong with sums')
        #         print(d)
        #         print(state['sums'])
        #         exit(0)


    def dp_gibbs_step():
        for ind in range(n_data):

            old_assignment = state['assignments'][ind]
            log_collapsed_assignment_conditional = [z_posterior_predictive(c,ind) + x_posterior_predictive(c,ind) for c in state['cluster_ids']+[-1]]

            # add this tweak to reduce underflow
            log_collapsed_assignment_conditional -= max(log_collapsed_assignment_conditional)
            log_collapsed_assignment_conditional = np.exp(log_collapsed_assignment_conditional)

            new_assignment = choice(state['cluster_ids']+[-1], p=log_collapsed_assignment_conditional/sum(log_collapsed_assignment_conditional))
            # update after assignment change

            if new_assignment == old_assignment:
                continue
            else:
                re_cluster(new_assignment, old_assignment, ind)


    def dp_gibbs():
        print('Dirichlet Process gibbs sampler')

        # update sum and size of each cluster
        for i,c in enumerate(state['assignments']):
            state['sizes'][c] += 1
            state['sums'][c] += fixed['data'][i]

        for i in range(train['n_burnin']):
            dp_gibbs_step()
            print('burn in | iter {}: {}'.format(i, state['assignments']))

        for i in range(train['n_iter']):
            dp_gibbs_step()
            sample['assignments'][i,:] = state['assignments']

            print('train | iter {}: {}'.format(i, state['assignments']))

            from collections import Counter
            if i % 10 == 0:
                plt.clf()
                argmax = [Counter(sample['assignments'][:i + 1, j].tolist()).most_common(1)[0][0] for j in range(n_data)]
                plt.scatter(data, y=[0] * len(data), c=argmax, s=10, cmap='gist_rainbow')
                plt.pause(0.05)

        plt.show()

    dp_gibbs()

if __name__ == '__main__':
    # make artificial data
    data = np.array(normal(1, 1, 20).tolist() + normal(10, 1, 20).tolist() + normal(20, 2, 20).tolist())
    plt.hist(data, bins=10)
    # plt.show()
    sample = DirichletMixtureModel(data, 60, n_clusters=10)
