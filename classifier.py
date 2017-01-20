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
        count = Counter(self.labels[i] for i in k_nearest_neighbors_indices)
        if count[0] > count[1]:
            return 0
        else:
            return 1


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
