from hw3_utils import abstract_classifier, abstract_classifier_factory
from euclidean_distance import euclidean_distance
from collections import Counter
from entropy import information_gain_for_continuous


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
            return self._train_aux({x for x in range(0, len(data[0]))}, static_classifier(0),
                                   data, labels, majority_factory(), idtree_classifier)

    # helper function to seperate data and labels
    def train_from_dataset(self, dataset):
        data = [x[:-1] for x in dataset]
        lables = [x[-1] for x in dataset]
        return self.train(data, lables)

    def _train_aux(self, Features, default, data, labels, leaf_classifier_factory, classifier):
        if len(labels) == 0:
            return classifier(None, None, None, None, default)

        c = leaf_classifier_factory.train(data, labels)

        if all(x == labels[0] for x in labels) or len(Features) == 0 or len(labels) <= self.L:
            return classifier(None, None, None, None, c)

        IG, f, v = self._selectFeature(Features, data, labels)
        if IG == 0:
            return classifier(None, None, None, None, c)

        f_domain = {e[f] for e in data}

        newFeatures = set(Features).discard(f) if len(f_domain) == 2 else Features

        index_l = [i for i in range(0, len(data)) if data[i][f] < v]
        index_r = [i for i in range(0, len(data)) if data[i][f] >= v]

        data_l = [data[i] for i in index_l]
        labels_l = [labels[i] for i in index_l]

        data_r = [data[i] for i in index_r]
        labels_r = [labels[i] for i in index_r]

        return classifier(f, v,
                          self._train_aux(newFeatures, default, data_l, labels_l, leaf_classifier_factory, classifier),
                          self._train_aux(newFeatures, default, data_r, labels_r, leaf_classifier_factory, classifier),
                          c)

    def _selectFeature(self, Features, data, labels):
        group = [data[i] + [labels[i]] for i in range(0, len(data))]
        best_IG = None
        for f in Features:
            IG, v = information_gain_for_continuous(group, f)
            if best_IG is None or IG > best_IG:
                best_IG = IG
                best_v = v
                best_f = f
        return best_IG, best_f, best_v


class static_classifier(abstract_classifier):
    def __init__(self, c):
        self.c = c

    def classify(self, features):
        return self.c


class majority_factory(abstract_classifier_factory):
    def train(self, data, labels):
        return static_classifier(Counter(labels).most_common()[0][0])


class enhanced_classifier(idtree_classifier):
    def __init__(self, feature, v, T_l, T_r, c):
        super().__init__(feature, v, T_l, T_r, c)


class enhanced_factory(idtree_factory):
    def __init__(self, L, k):
        self.L = L
        self.k = k

    def train(self, data, labels):
        if len(data) == 0:
            return enhanced_classifier(None, None, None, None, static_classifier(0))
        else:
            return self._train_aux({x for x in range(0, len(data[0]))}, static_classifier(0), data,
                                   labels, knn_factory(self.k), enhanced_classifier)
