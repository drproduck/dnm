# Processes as iterators
from random import *
from numpy.random import beta
class Process():
    def __iter__(self):
        return self
    def next(self):
        return sample(1)
    def getWeights(self):
        return self.weights
    def sample(self,K):
        pass
    
    
class FakeProcess(Process):
    def __init__(self,alpha):
        pass
        
        
class StickBreakingProcess(Process):
    """Stick breaking with discrete base"""
    def __init__(self, alpha, K=100):
        self.alpha = alpha
        self.weights = []
        self.current_cum = 0
        self.max_atom = -1
        # init with K=100 to save time
        if K < 100: raise Exception('must initialize with K >= 100')
            
        bt = beta(1,self.alpha,K)
        # first weight is bt[0]
        self.weights[0].append(bt[0])
        self.prod = (1 - bt[0])
        self.current_cum += bt[0]
        for i in range(1,K):
            self.weights[i].append(bt[i]*self.prod)
            self.current_cum += bt[i]*self.prod
            self.prod *= (1 - bt[i])

        self.weights.append(1 - self.current_sum)
        
    def sample(self, K):
        
        res = np.zeros(K)
        
        for i in range(K):
            next_val = choices(range(len(self.weights)), weights=self.weights)
        
            # if val is less than current available weights but more than last seen atom
            if next_val < M and next_val > self.max_atom:
                self.max_atom += 1
                res[i] = self.max_atom
                
            elif next_val <= self.max_atom:
                res[i] = next_val
                
            # if val is from unseen weight
            elif next_val == M:

                bt = beta(1,self.alpha)
                self.current_cum += bt[i]*self.prod
                self.weights[-1] = bt[i]*self.prod
                
                self.prod *= (1-bt)
                
                self.weights.append(1 - self.current_cum)
                
                self.max_atom += 1
                res[i] = self.max_atom
                
        return res
                
        
        
      
                

        