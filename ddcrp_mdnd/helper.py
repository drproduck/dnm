from scipy.special import expit


def linear_distance(i, j):
    """used for timestamp"""
    return i - j


def window_decay(w=1):
    def f(x):
        return x <= w
    return f


def logistic_decay(a, b=1):
    def f(x):
        return expit((-x+a)/b)
    return f


def identity_decay():
    def f(x): return 1
    return f
# def distance_matrix(n, distance, alpha):
