from numpy import linalg as LA


def euclidean_distance(v0, v1):
    return LA.norm([a - b for a, b in zip(v0, v1)])
