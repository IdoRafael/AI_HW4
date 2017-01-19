from cross_validation import split_crosscheck_groups
from hw3_utils import create_dataset
from collections import Counter

def compare(s, t):
    t = list(t)   # make a mutable copy
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False
    return not t

if __name__ == '__main__':
    d = create_dataset()
    d1 = create_dataset('ecg_fold_1.data')
    d2 = create_dataset('ecg_fold_2.data')
    d3 = d1 + d2

    print(compare(d,[1,2]))
