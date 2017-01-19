from numpy import linalg as LA
from math import sqrt


def euclidean_distance(v0, v1):
    return sqrt(sum([(v0[i] - v1[i])**2 for i in range(0,len(v0))]))


# faster
def euclidean_distance1(v0, v1):
    return LA.norm([(v0[i] - v1[i]) for i in range(0,len(v0))])

