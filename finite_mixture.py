import numpy as np
from numpy.random import choice, dirichlet, normal
import scipy.stats as stats
import matplotlib.pyplot as plt


def GaussianMixture(data, n_data, n_clusters, scale=0.1):
    """Bayes GMM with fixed sigma
    Using Blei's bayesian graphical model"""

    n_clusters = n_clusters
    state = {
        'assignments': np.array([choice(n_clusters, replace=True) for _ in range(n_data)], dtype=int),
        'means': np.random.randn(n_clusters) * 2,
        'pi': np.random.dirichlet(alpha=[2] * 3),
        'sizes': [0] * n_clusters,
        'sums': [0] * n_clusters
    }

    fixed = {
        'sigma_data': 1,
        'sigma_mean': 1,
        'mean_mean': 1,
        'alpha_pi': 5,
        'data': data
    }

    train = {
        'n_iter': 50,
        'n_burnin': 10,
    }

    sample = {
        'assignments': np.zeros((train['n_iter'], n_data), dtype=int),
        'means': np.zeros((train['n_iter'], n_clusters)),
        'pi': np.zeros((train['n_iter'], n_clusters))
    }

    # p(z_i = k|.)
    def log_assignment_score(data, cluster):
        return np.log(state['pi'][cluster]) + stats.norm.logpdf(data, state['means'][cluster], fixed['sigma_data'])

    def sample_assignment():
        """calculate probs for each z, and sample a new assignment"""
        for i, x in enumerate(data):
            probs = np.exp([log_assignment_score(x, z) for z in range(n_clusters)])
            state['assignments'][i] = choice(n_clusters, p=probs / sum(probs))

        # update size of each cluster
        state['sizes'] = [0] * n_clusters
        for i in state['assignments']:
            state['sizes'][i] += 1

    def sample_pi():
        """sample new mixture weights"""
        alpha = [state['sizes'][i] + fixed['alpha_pi'] for i in range(n_clusters)]
        state['pi'] = dirichlet(alpha=alpha, size=1).flatten()

    def sample_means():
        """sample new cluster means"""
        cluster_sums = [sum(fixed['data'][state['assignments'] == c]) for c in range(n_clusters)]
        mu_mean = [
            fixed['sigma_mean'] * cluster_sums[c] / (fixed['sigma_data'] + fixed['sigma_mean'] * state['sizes'][c]) for
            c in range(n_clusters)]
        mu_var = [1 / (state['sizes'][c] / fixed['sigma_data'] + 1 / fixed['sigma_mean']) for c in range(n_clusters)]
        state['means'] = [normal(mu_mean[c], mu_var[c]) for c in range(n_clusters)]

    def gibbs_step():
        sample_assignment()
        sample_pi()
        sample_means()

    def gibbs():
        print('gibbs sampler')

        for i in range(train['n_burnin']):
            gibbs_step()
            print('burn in | iter {}: {}'.format(i, state['assignments']))

        for i in range(train['n_iter']):
            gibbs_step()

            sample['assignments'][i, :] = state['assignments']
            sample['means'][i, :] = state['means']
            sample['pi'][i, :] = state['pi']

            print('train | iter {}: {}'.format(i, state['assignments']))

            from collections import Counter
            if i % 10 == 0:
                plt.clf()
                argmax = [Counter(sample['assignments'][:i + 1, j].tolist()).most_common(1)[0][0] for j in range(n_data)]
                plt.scatter(data, y=[0] * len(data), c=argmax, s=10)
                plt.pause(0.05)

        plt.show()

    #================================================================================================
    #collapsed gibbs starts here

    def assignment_posterior_predictive(cluster, removed_data_ind):
        """p(z_i=k|z_{-i}"""
        if state['assignments'][removed_data_ind] == cluster:
            Nk = state['sizes'][cluster] - 1
        else:
            Nk = state['sizes'][cluster]
        return np.log(Nk + fixed['alpha_pi']) - np.log(n_data - 1 + fixed['alpha_pi'])

    def data_posterior_predictive(cluster, removed_data_ind):
        if state['assignments'][removed_data_ind] == cluster:
            Nk = state['sizes'][cluster] - 1
            sumk = state['sums'][cluster]
        else:
            Nk = state['sizes'][cluster]
            sumk = state['sums'][cluster]
        sigk = 1 / (Nk / fixed['sigma_data'] + 1 / fixed['sigma_mean'])
        muk = sigk * (fixed['mean_mean'] / fixed['sigma_mean'] + sumk / fixed['sigma_data'])

        return stats.norm.logpdf(fixed['data'][removed_data_ind], muk, np.sqrt(sigk))

    def collapsed_gibbs_step():
        for ind in range(n_data):
            old_assignment = state['assignments'][ind]
            collapsed_assignment_conditional = [assignment_posterior_predictive(c,ind) + data_posterior_predictive(c,ind) for c in range(n_clusters)]

            #add this tweak to reduce underflow
            collapsed_assignment_conditional -= max(collapsed_assignment_conditional)
            collapsed_assignment_conditional = np.exp(collapsed_assignment_conditional)

            new_assignment = choice(n_clusters, p=collapsed_assignment_conditional/sum(collapsed_assignment_conditional))

            #update after assignment change
            state['assignments'][ind] = new_assignment

            state['sums'][old_assignment] -= fixed['data'][ind]
            state['sums'][new_assignment] += fixed['data'][ind]
            state['sizes'][old_assignment] -= 1
            state['sizes'][new_assignment] += 1



    def collapsed_gibbs():
        print('collapsed gibbs sampler')

        # update sum and size of each cluster
        for i,c in enumerate(state['assignments']):
            state['sizes'][c] += 1
            state['sums'][c] += fixed['data'][i]

        for i in range(train['n_burnin']):
            collapsed_gibbs_step()
            print('burn in | iter {}: {}'.format(i, state['assignments']))

        for i in range(train['n_iter']):
            collapsed_gibbs_step()
            sample['assignments'][i,:] = state['assignments']

            print('train | iter {}: {}'.format(i, state['assignments']))

            from collections import Counter
            if i % 10 == 0:
                plt.clf()
                argmax = [Counter(sample['assignments'][:i + 1, j].tolist()).most_common(1)[0][0] for j in range(n_data)]
                plt.scatter(data, y=[0] * len(data), c=argmax, s=10)
                plt.pause(0.05)

        plt.show()

    # gibbs()
    collapsed_gibbs()

if __name__ == '__main__':
    # make artificial data
    data = np.array(normal(1, 1, 50).tolist() + normal(2, 1, 50).tolist() + normal(4, 2, 50).tolist())
    plt.hist(data, bins=10)
    # plt.show()
    sample = GaussianMixture(data, 150, n_clusters=3)
