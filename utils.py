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


class UnionFind:
    def __init__(self):
        '''\
Create an empty union find data structure.'''
        self.num_weights = {}
        self.parent_pointers = {}
        self.num_to_objects = {}
        self.objects_to_num = {}
        self.__repr__ = self.__str__

    def insert_objects(self, objects):
        """\
Insert a sequence of objects into the structure.  All must be Python hashable."""
        for object in objects:
            self.find(object)

    def find(self, object):
        """\
Find the root of the set that an object is in.
If the object was not known, will make it known, and it becomes its own set.
Object must be Python hashable."""
        if not object in self.objects_to_num:
            obj_num = len(self.objects_to_num)
            self.num_weights[obj_num] = 1
            self.objects_to_num[object] = obj_num
            self.num_to_objects[obj_num] = object
            self.parent_pointers[obj_num] = obj_num
            return object
        stk = [self.objects_to_num[object]]
        par = self.parent_pointers[stk[-1]]
        while par != stk[-1]:
            stk.append(par)
            par = self.parent_pointers[par]
        for i in stk:
            self.parent_pointers[i] = par
        return self.num_to_objects[par]

    def union(self, object1, object2):
        """\
Combine the sets that contain the two objects given.
Both objects must be Python hashable.
If either or both objects are unknown, will make them known, and combine them."""
        o1p = self.find(object1)
        o2p = self.find(object2)
        if o1p != o2p:
            on1 = self.objects_to_num[o1p]
            on2 = self.objects_to_num[o2p]
            w1 = self.num_weights[on1]
            w2 = self.num_weights[on2]
            if w1 < w2:
                o1p, o2p, on1, on2, w1, w2 = o2p, o1p, on2, on1, w2, w1
            self.num_weights[on1] = w1 + w2
            del self.num_weights[on2]
            self.parent_pointers[on2] = on1


if __name__ == '__main__':
    # p = np.array([2, 3, 4])
    # p, _ = log_normalize(p)
    # print(np.exp(p))
    # print(log_choice(p))

    dj = UnionFind()
    dj.union(1, 2)
    dj.union(2, 3)
    dj.union(4, 5)
    print(dj.parent_pointers)
    dj.parent_pointers.pop([0, 1], None)
    # print(dj.num_to_objects)
    # print(dj.objects_to_num)

