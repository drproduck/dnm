# Processes as iterators
from random import choices
from numpy.random import *
import numpy as np
from scipy.stats import norm


# class Process:
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         return sample(1)
#
#     def getWeights(self):
#         return self.weights
#
#     def sample(self):
#         pass
#


class RandomVariable():
    """wrapper for some distributions"""

    def __init__(self):
        self.rv = None

    # def sample(self):
    #     return self.rv.rvs()


class Gaussian(RandomVariable):
    def __init__(self, mu, sigma):
        self.rv = norm(mu, sigma)

    def sample(self):
        return self.rv.rvs()


class StickBreakingProcess():
    """Stick breaking with discrete base"""

    def __init__(self, ap, K=100):
        self.ap = ap
        self.weights = []
        self.current_cum = 0
        self.max_atom = -1
        # init with K=100 to save time
        if K < 100: raise Exception('must initialize with K >= 100')

        bt = beta(1, self.ap, K)
        # first weight is bt[0]
        self.weights.append(bt[0])
        self.prod = (1 - bt[0])
        self.current_cum += bt[0]
        for i in range(1, K):
            self.weights.append(bt[i] * self.prod)
            self.current_cum += bt[i] * self.prod
            self.prod *= (1 - bt[i])

        self.weights.append(1 - self.current_cum)

    def sample(self, K=1):

        res = np.zeros(K, dtype=np.int32)

        for i in range(K):
            next_val = choices(range(len(self.weights)), weights=self.weights)[0]

            # if val is less than current available weights but more than last seen atom
            if len(self.weights) - 1 > next_val > self.max_atom:
                self.max_atom += 1
                res[i] = self.max_atom

            elif next_val <= self.max_atom:
                res[i] = next_val

            # if val is from unseen weight (rarely)
            elif next_val == len(self.weights) - 1:

                bt = beta(1, self.ap)
                self.current_cum += bt * self.prod
                self.weights[-1] = bt * self.prod

                self.prod *= (1 - bt)

                self.weights.append(1 - self.current_cum)

                self.max_atom += 1
                res[i] = self.max_atom
        return res

    def sample(self):
        return sample(K=1)

class DirichletProcessDiscrete():
    """dirichlet Process with uniform base measure {0,1,2...}"""

    def __init__(self, ap=1):
        self.ap = ap
        self.next_value = 0
        self.weights = []
        self.remaining = 1

    def sample(self):

        def return_new_val_and_update():
            # pick new value
            s = self.next_value
            self.next_value += 1

            # update spec
            bt = beta(1, self.ap)
            self.weights.append(self.remaining * bt)
            self.remaining *= (1 - bt)

            return s

        # first draw
        if self.next_value == 0:
            return return_new_val_and_update()

        else:
            # pick from past values or from base?
            roll = choice(range(len(self.weights)+1), p=self.weights + [self.remaining])

            # if unseen
            if roll == len(self.weights):
                return return_new_val_and_update()

            elif roll < len(self.weights):
                # pick past value
                return roll

class DirichletProcess():
    """Dirichlet Process with a more general base distribution """

    def __init__(self, base,ap=1):
        self.ap = ap
        self.base = base
        self.vals = []
        self.weights = []
        self.remaining = 1

    def sample(self):

        def return_new_val_and_update():
            # pick new value
            s = self.base.sample()
            self.vals.append(s)
            # update spec
            bt = beta(1, self.ap)
            self.weights.append(self.remaining * bt)
            self.remaining *= (1 - bt)
            return s

        # first draw
        if len(self.vals) == 0:
            return return_new_val_and_update()

        else:
            # pick from past values or from base?
            roll = choice(range(len(self.weights)+1), p=self.weights + [self.remaining])

            # if unseen
            if roll == len(self.weights):
                return return_new_val_and_update()

            elif roll < len(self.weights):
                # pick past value
                return self.vals[roll]


class HierarchicalDirichletProcess():

    def __init__(self, ap1, ap2, base):
        baseDP = DirichletProcess(ap1, base)
        self.DP = DirichletProcess(ap2, baseDP)

    def sample(self):
        return self.DP.sample()




class Antoniak():
    """Antoniak Distribution with a cache for lazy evaluating stirling numbers"""

    def __init__(self):
        self.cache = [] # cache
        self.maxnn = 1

    def stirling(self,nn): # making an array for keep the stirling(N,1:N) for saving time consumming
        if len(self.cache)==0:
            self.cache.append([])
            self.cache[0].append(1)
        if nn > self.maxnn:
            for mm in range (self.maxnn,nn):
                ln=len(self.cache[mm-1])+1
                self.cache.append([])

                for xx in range(ln) :
                    self.cache[mm].append(0)
                    if xx< (ln-1):
                        self.cache[mm][xx] += self.cache[mm-1][xx]*mm
                    if xx>(ln-2) :
                        self.cache[mm][xx] += 0
                    if xx==0 :
                        self.cache[mm][xx] += 0
                    if xx!=0 :
                        self.cache[mm][xx] += self.cache[mm-1][xx-1]

            self.maxnn=nn
        return self.cache[nn-1]

    def sample(self, alpha, n):
        ss = self.stirling(n)
        max_val = max(ss)
        p = np.array(ss) / max_val

        aa = 1
        for i, _ in enumerate(p):
            p[i] *= aa
            aa *= alpha

            p = np.array(p,dtype='float') / np.array(p,dtype='float').sum()
            return choice(range(1, n+1), p=p)


if __name__ == '__main__':
    # A = StickBreakingProcess(ap=1)
    # B = DirichletProcess(base=A, ap=1)
    # print(B.sample())

    a = Antoniak()
    print(a.stirling(3))
    print(a.sample(alpha=0.2, n=10))
    # print(a.cache)
