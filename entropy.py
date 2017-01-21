from collections import Counter, defaultdict
from math import log2


def entropy(group):
    if not group:
        return 0
    counter = Counter(group)
    return sum(-(counter[i] / len(group)) * log2(counter[i] / len(group)) for i in counter)


def information_gain(group, f):
    if not group:
        return 0
    E = [x[-1] for x in group]
    E_i = {v: [e[-1] for e in group if e[f] == v] for v in {e[f] for e in group}}
    return entropy(E) - sum((len(E_i[v]) / len(E)) * entropy(E_i[v]) for v in E_i)


def information_gain_for_continuous(group, f):
    if not group:
        return 0, None
    IG = {v: information_gain([_change_value(e, f, 0) if e[f] < v
                               else _change_value(e, f, 1) for e in group], f)
          for v in {e[f] for e in group}}
    best_v = max(IG, key=(lambda key: IG[key]))
    return IG[best_v], best_v


def _change_value(e, f, v):
    new_e = list(e)
    new_e[f] = v
    return new_e





