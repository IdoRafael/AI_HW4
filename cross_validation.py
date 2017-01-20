from random import shuffle
from hw3_utils import create_dataset


def split_crosscheck_groups(dataset, num_folds):
    shuffed_dataset = list(dataset)
    shuffle(shuffed_dataset)

    list0 = [o for o in shuffed_dataset if o[-1] == 0]
    list1 = [o for o in shuffed_dataset if o[-1] == 1]

    size_list0 = len(list0) // num_folds
    size_list1 = len(list1) // num_folds

    gen0 = (list0[i:i + size_list0] for i in range(0, len(list0) - (len(list0) % num_folds), size_list0))
    gen1 = (list1[i:i + size_list1] for i in range(0, len(list1) - (len(list1) % num_folds), size_list1))

    for i in range(1, num_folds + 1):
        current_fold = next(gen0) + next(gen1)
        with open('ecg_fold_{}.data'.format(i), 'w') as f:
            for line in current_fold:
                f.write(', '.join([str(x) for x in line]) + '\n')


def evaluate(classifier_factory, k):
    folds = [create_dataset('ecg_fold_{}.data'.format(i)) for i in range(1, k + 1)]

    total_accuracy = 0
    total_error = 0

    for i in range(0, k):
        accuracy = 0
        error = 0
        train_set = [folds[j] for j in range(0, k) if j != i]
        classifier = classifier_factory.train_from_dataset([item for sublist in train_set for item in sublist])

        for features in folds[i]:
            if classifier.classify(features[:-1]) == features[-1]:
                accuracy += 1
            else:
                error += 1

        accuracy /= len(folds[i])
        error /= len(folds[i])

        total_accuracy += accuracy
        total_error += error
    return total_accuracy / k, total_error / k
