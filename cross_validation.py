from random import shuffle


def split_crosscheck_groups(dataset, num_folds):
    shuffle(list(dataset))

    list0 = [o for o in dataset if o[-1] == 0]
    list1 = [o for o in dataset if o[-1] == 1]

    size_list0 = len(list0) // num_folds
    size_list1 = len(list1) // num_folds

    gen0 = (list0[i:i + size_list0] for i in range(0, len(list0) - (len(list0) % num_folds), size_list0))
    gen1 = (list1[i:i + size_list1] for i in range(0, len(list1) - (len(list1) % num_folds), size_list1))

    for i in range(1, num_folds + 1):
        current_fold = next(gen0) + next(gen1)
        with open('ecg_fold_{}.data'.format(i), 'w') as f:
            for line in current_fold:
                f.write(', '.join([str(x) for x in line]) + '\n')

