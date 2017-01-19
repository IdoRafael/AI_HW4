from random import shuffle

if __name__ == '__main__':
    x = [[i, 0] for i in range(5)] + [[i, 1] for i in range(7)]

    l0 = [o for o in x if o[-1] == 0]
    l1 = [o for o in x if o[-1] == 1]

    shuffle(l0)
    shuffle(l1)

    k = 4
    n0 = len(l0) // k
    n1 = len(l1) // k

    gen0 = (l0[i:i + n0] for i in range(0, len(l0) - (len(l0) % k), n0))
    gen1 = (l1[i:i + n1] for i in range(0, len(l1) - (len(l1) % k), n1))

    for l in gen0:
        print(l)

    for l in gen1:
        print(l)


