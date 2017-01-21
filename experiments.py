from cross_validation import evaluate, split_crosscheck_groups
from classifier import knn_factory, idtree_factory, enhanced_factory
from hw3_utils import create_dataset
import csv


def run_experiment6():
    with open('experiments6.csv', 'w', newline='') as f:
        a = csv.writer(f, delimiter=',')
        for k in [1, 3, 5, 7, 13]:
            accuracy, error = evaluate(knn_factory(k), 2)
            a.writerow([k, accuracy, error])


def run_experiment9():
    with open('experiments9.csv', 'w', newline='') as f:
        a = csv.writer(f, delimiter=',')
        for L in [1, 7, 13, 17, 21]:
            accuracy, error = evaluate(idtree_factory(L), 2)
            a.writerow([L, accuracy, error])


def run_experiment12():
    with open('experiments12.csv', 'w', newline='') as f:
        a = csv.writer(f, delimiter=',')
        for L in [1, 7, 13, 17, 21]:
            for k in [1, 3, 5, 7, 13]:
                accuracy, error = evaluate(enhanced_factory(L, k), 2)
                a.writerow([L, k, accuracy, error])


if __name__ == '__main__':
    # split_crosscheck_groups(create_dataset(), 2)
    run_experiment6()
    run_experiment9()
    run_experiment12()
