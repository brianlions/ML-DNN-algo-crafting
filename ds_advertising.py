#!/usr/bin/env python3
import numpy as np

def load_data_set(one_tenth=False):
    filename = './datasets/ISLR2/Advertising.csv'
    with open(filename) as fh:
        records = []
        for line_num, line in enumerate(fh):
            if line_num > 0:
                records.append([float(f) for f in line.strip().split(',')])
        records = np.array(records, dtype=float)
        n_records = len(records)
        if n_records != 200:
            raise Exception('wrong num of records: {} != 200'.format(n_records))

        np.take(records, np.random.permutation(n_records), axis=0, out=records)
        if not one_tenth:
            pos1, pos2, pos3 = 140, 170, n_records
        else:
            pos1, pos2, pos3 = 14, 17, 20
        features = records[:, 1:4] # TV, radio, newspaper
        targets  = records[:, 4]   # sales
        train_f, eval_f, test_f = features[:pos1], features[pos1:pos2], features[pos2:pos3]
        train_t, eval_t, test_t = targets[:pos1],  targets[pos1:pos2],  targets[pos2:pos3]

        return (train_f, train_t), (eval_f, eval_t), (test_f, test_t)
