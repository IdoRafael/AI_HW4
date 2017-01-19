from numpy import linalg as LA
from math import sqrt


# fastest
def euclidean_distance(v0, v1):
    return LA.norm([a - b for a, b in zip(v0, v1)])


# faster
def euclidean_distance1(v0, v1):
    return LA.norm([(v0[i] - v1[i]) for i in range(0,len(v0))])


def euclidean_distance2(v0, v1):
    return sqrt(sum([(v0[i] - v1[i])**2 for i in range(0,len(v0))]))
