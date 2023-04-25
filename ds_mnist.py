#!/usr/bin/env python3

import gzip
import logging
import os
import random
import struct
import numpy as np

NO_SHUFFLE    = 0
SHUFFLE_TRAIN = 1
SHUFFLE_ALL   = 2

class MnistRawReader(object):
    def __init__(self, shuffle_mode = NO_SHUFFLE, one_tenth=False):
        if shuffle_mode not in (NO_SHUFFLE, SHUFFLE_TRAIN, SHUFFLE_ALL):
            raise Exception('invalid mnist shuffle mode: %d' % shuffle_mode)
        self._shuffle_mode = shuffle_mode
        self._one_tenth = one_tenth
        self._load()

    def _read_raw_file(self, filename, head_len, elem_len, limit=0):
        logging.info('loading mnist data file: {}{}'.format(filename,
            limit > 0 and ', limit {}'.format(limit) or ''))
        with open(filename, 'rb') as fh:
            fh.read(head_len)
            data_len = os.stat(filename).st_size - head_len
            if limit > 0:
                data_len = min(data_len, limit * elem_len)
            data = fh.read(data_len)
            return [data[pos:pos+elem_len] for pos in range(0, data_len, elem_len)]

    def _load(self):
        if self._one_tenth:
            logging.warning('using only 1 / 10 of all available data')
        if not self._one_tenth:
            images = self._read_raw_file('./datasets/mnist/train-images-idx3-ubyte', 16, 28 * 28) \
                   + self._read_raw_file('./datasets/mnist/t10k-images-idx3-ubyte',  16, 28 * 28)
            labels = self._read_raw_file('./datasets/mnist/train-labels-idx1-ubyte', 8,  1) \
                   + self._read_raw_file('./datasets/mnist/t10k-labels-idx1-ubyte',  8,  1)
        else:
            images = self._read_raw_file('./datasets/mnist/train-images-idx3-ubyte', 16, 28 * 28, 7000)
            labels = self._read_raw_file('./datasets/mnist/train-labels-idx1-ubyte', 8,  1, 7000)

        def buffer2array(data):
            if 1:
                img = [(pixel >= 128 and 1.0 or 0.0) for pixel in struct.unpack('B' * 28 * 28, data)]
            else:
                img = [1.0 * pixel / 256 for pixel in struct.unpack('B' * 28 * 28, data)]
            return img

        self._images = np.array([buffer2array(img) for img in images], dtype=float)
        self._labels = np.array([ord(label) for label in labels], dtype=int)

        if self._shuffle_mode == SHUFFLE_ALL:
            shuffle_len = self._one_tenth and 7000 or 70000
        elif self._shuffle_mode == SHUFFLE_TRAIN:
            shuffle_len = self._one_tenth and 6000 or 60000
        else:
            shuffle_len = 0

        if shuffle_len:
            orders = np.random.permutation(shuffle_len)
            np.take(self._images, orders, axis=0, out=self._images[:shuffle_len])
            np.take(self._labels, orders, axis=0, out=self._labels[:shuffle_len])

        def distribution(name, labels):
            counter = np.zeros(10, dtype=int)
            for y in labels:
                counter[y] += 1
            logging.info('{:s} [{:s}], total {:d}'.format(name,
                ' '.join(['{:4d}'.format(c) for c in counter]), np.sum(counter)))

        train_end = self._one_tenth and 5000 or 50000
        valid_end = self._one_tenth and 6000 or 60000
        distribution('training dataset:  ', self._labels[ : train_end])
        distribution('validation dataset:', self._labels[train_end : valid_end])
        distribution('testing dataset:   ', self._labels[valid_end : ])

    def data(self):
        return self._images, self._labels

    def dump_image_idx(self, index, bg='.'):
        buf = []
        image = self._images[index]
        for pos in range(0, 28 * 28, 28):
            buf.append(''.join(p > 0.5 and 'x' or bg for p in image[pos:pos+28]))
        print('----------\nlabel: %d\n%s\n' % (self._labels[index], '\n'.join(buf)))

def dump_one(image, bg='.'):
    buf = []
    for pos in range(0, 28 * 28, 28):
        buf.append(''.join(p > 0.5 and 'x' or bg for p in image[pos:pos+28]))
    print('\n'.join(buf))

def load_data_set(shuffle_mode = NO_SHUFFLE, one_tenth=False):
    images, labels = MnistRawReader(shuffle_mode, one_tenth).data()
    pos1 = one_tenth and 5000 or 50000
    pos2 = one_tenth and 6000 or 60000
    tr_imgs, va_imgs, te_imgs = images[:pos1], images[pos1:pos2], images[pos2:]
    tr_lbls, va_lbls, te_lbls = labels[:pos1], labels[pos1:pos2], labels[pos2:]
    return (tr_imgs, tr_lbls), (va_imgs, va_lbls), (te_imgs, te_lbls)

if __name__ == '__main__':
    import log_setup
    log_setup.setup()
    reader = MnistRawReader(1, one_tenth=1)
    for i in range(2):
        reader.dump_image_idx(i)
