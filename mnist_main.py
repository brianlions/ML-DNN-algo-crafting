#!/usr/bin/env python3
import sys
import logging
import numpy as np
import ds_mnist
import dnn
import log_setup

if __name__ == '__main__':
    log_setup.setup()

    if len(sys.argv) not in (2, 3):
        print('''usage: %s 1|2|3 [model.json]
                1 train + save
                2 load + eval
                3 train + save + load + eval
                ''' % sys.argv[0])
        sys.exit(1)

    cmd = int(sys.argv[1])
    model_file = 'model_mnist.json'
    if len(sys.argv) > 2:
        model_file = sys.argv[2]

    one_tenth = 0 # uses 1/10 of mnist data

    logging.info('loading image data ...')
    if 0:
        train_data, validation_data, test_data = ds_mnist.load_data_set(ds_mnist.NO_SHUFFLE, one_tenth=one_tenth)
    else:
        train_data, validation_data, test_data = ds_mnist.load_data_set(ds_mnist.SHUFFLE_ALL, one_tenth=one_tenth)

    def train_nn_v2():
        net = dnn.NeuralNetwork(input_size=784, fc_sizes=[30, 10], lr=0.03, loss='ce',
                #l2reg=5,
                #momentum=0.7,
                l2reg=1e-3,
                #l1reg=3,
                #dropout=0.2,
                activations=['relu', 'sigmoid'],
                stride=8)
        net.train(train_data, epochs=5, batch_size=64,
                eval_data=validation_data, collect_eval_stat=True, collect_train_stat=True)
        net.save(model_file)
        return net

    def load_n_eval(max_show = 3):
        net = dnn.NeuralNetwork.load(model_file)
        result = net.evaluate(test_data[0], test_data[1], label_is_matrix=False)
        n_correct = np.sum(result[:,0] == result[:,1])
        logging.info('test data set: accuracy %5d / %5d = %.4f' % (
            n_correct, len(test_data[0]), 1.0 * n_correct / len(test_data[0])))
        if max_show <= 0:
            return net

        n = 0
        for img, l, p in list(zip(test_data[0], test_data[1], result)):
            if p[0] != l and n < max_show:
                print('# %d - predict: %d, actual: %d' % (n + 1, p[0], l))
                ds_mnist.dump_one(img)
                n += 1

        return net

    if cmd == 1:
        net = train_nn_v2()
    elif cmd == 2:
        net = load_n_eval(20)
    elif cmd == 3:
        train_nn_v2()
        load_n_eval(20)
    else:
        logging.error('invalid command: %d' % cmd)
        sys.exit(1)
