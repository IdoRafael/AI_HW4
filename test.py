import cross_validation
import classifier
from hw3_utils import create_dataset
from random import shuffle
from collections import Counter
from entropy import entropy, information_gain, information_gain_for_continuous


def compare_lists_without_order(s, t):
    t = list(t)   # make a mutable copy
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False
    return not t


if __name__ == '__main__':
    group = [[1,1]] + [[0,0] for i in range(0,100)] +[[2,1] for i in range(0,100)]
    f = 0
    print(information_gain_for_continuous(group, f))

