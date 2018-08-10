"""some utilities function"""
import numpy as np


def log_normalize(v):
    """ return log(sum(exp(v)))"""

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v) + 1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1] + 1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:, np.newaxis]

    return v, log_norm


def log_choice(lp):
    """

    :param lp: log prob (normalized)
    :return: index in range(len(lp))
    """
    return np.searchsorted(np.cumsum(np.exp(lp)), np.random.random())


if __name__ == '__main__':
    p = np.array([2, 3, 4])
    p, _ = log_normalize(p)
    print(np.exp(p))
    print(log_choice(p))
