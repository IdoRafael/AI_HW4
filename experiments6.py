from cross_validation import evaluate, split_crosscheck_groups
from classifier import knn_factory
from hw3_utils import create_dataset
import csv


def run_experiment():
    with open('experiments6.csv', 'w', newline='') as f:
        a = csv.writer(f, delimiter=',')
        for k in [1, 3, 5, 7, 13]:
            accuracy, error = evaluate(knn_factory(k), 2)
            a.writerow([k, accuracy, error])


if __name__ == '__main__':
    split_crosscheck_groups(create_dataset(), 2)
    run_experiment()