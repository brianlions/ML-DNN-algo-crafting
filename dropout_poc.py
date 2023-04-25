#!/usr/bin/env python3

import numpy as np

def build_masks(sizes, dropout=0.7):
    print('sizes:', sizes)
    weights = [10 * np.ones((1, y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
    print('w:')
    for w in weights:
        print(w)
    print('m:')
    masks = [np.ones((1, y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
    for m in masks:
        print(m)
    pos = [np.random.binomial(1, dropout, sizes[1:-1])]
    print('p:', pos)

    print('-----------')
    for l, p in enumerate(pos):
        masks[l] *= p.reshape((-1,1))
        masks[l+1] *= p.reshape((1,-1))
    for m in masks:
        print(m)
    print()
    for w, m in zip(weights, masks):
        print(w * m)

if __name__ == '__main__':
    #build_masks([784, 50, 30, 10], 0.3)
    build_masks([10, 7, 4])
