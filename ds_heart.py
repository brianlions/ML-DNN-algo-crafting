#!/usr/bin/env python3
import numpy as np

def load_dataset():
    filename = './datasets/ISLR2/Heart.csv'
    with open(filename) as fh:
        feat_names = []
        idx2map = { 2: {}, 12: {}, 13: {} }

        records = []
        for line_num, line in enumerate(fh):
            if line_num == 0:
                feat_names = [f.strip('"') for f in line.strip().split(',')[1:]]
                continue

            fields = line.strip().split(',')[1:]
            if 'NA' in fields:
                continue
            for idx in (2, 12, 13):
                str_val = fields[idx].strip('"')
                if str_val not in idx2map[idx]:
                    idx2map[idx][str_val] = len(idx2map[idx])
                fields[idx] = idx2map[idx][str_val]
            records.append(fields)

        return feat_names, np.array(records, dtype=float), idx2map

if __name__ == '__main__':
    res = load_dataset()
    print(res[0])
    print(res[2])
    print(res[1], len(res[1]))
