from hw3_utils import abstract_classifier, abstract_classifier_factory
from euclidean_distance import euclidean_distance
from collections import Counter


class knn_classifier(abstract_classifier):
    def __init__(self, k, data, labels):
        self.k = k
        self.data = data
        self.labels = labels

    def classify(self, features):
        k_nearest_neighbors_indices = \
            sorted(range(len(self.data)), key=lambda x: euclidean_distance(self.data[x], features))[:self.k]
        return Counter(self.labels[i] for i in k_nearest_neighbors_indices).most_common()[0][0]


class knn_factory(abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(self.k, data, labels)

    # helper function to seperate data and labels
    def train_from_dataset(self, dataset):
        data = [x[:-1] for x in dataset]
        lables = [x[-1] for x in dataset]
        return self.train(data, lables)


class idtree_classifier(abstract_classifier):
    def __init__(self, feature, v, T_l, T_r, c):
        self.feature = feature
        self.v = v
        self.T_l = T_l
        self.T_r = T_r
        self.c = c

    def classify(self, features):
        if self.T_l is None and self.T_r is None:
            return self.c.classify(features)
        return (self.T_l if features[self.feature] < self.v else self.T_r).classify(features)


class idtree_factory(abstract_classifier_factory):
    def __init__(self, L):
        self.L = L

    def train(self, data, labels):
        if len(data) == 0:
            return idtree_classifier(None, None, None, None, static_classifier(0))
        else:
            return self.train_aux([x for x in range(0, len(data[0]))], static_classifier(0), data, labels)

    # helper function to seperate data and labels
    def train_from_dataset(self, dataset):
        data = [x[:-1] for x in dataset]
        lables = [x[-1] for x in dataset]
        return self.train(data, lables)

    def train_aux(self, Features, default, data, labels):
        if len(labels) == 0:
            return idtree_classifier(None, None, None, None, default)


class static_classifier(abstract_classifier):
    def __init__(self, c):
        self.c = c

    def classify(self, features):
        return self.c


class majority_classifier(static_classifier):
    def __init__(self, labels):
        self.c = Counter(labels).most_common()[0][0]
