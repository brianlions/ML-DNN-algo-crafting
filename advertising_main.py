#!/usr/bin/env python3
import logging
import sys
import dnn
import ds_advertising as ds_ads
import log_setup

log_setup.setup()
#log_setup.reset_level('debug')
train_data, eval_data, test_data = ds_ads.load_data_set(one_tenth=0)
logging.info('{} {} {}'.format(len(train_data[0]), len(eval_data[0]), len(test_data[0])))
net = dnn.NeuralNetwork(input_size=3, fc_sizes=[1], lr=1e-5, loss='mse', activations=['linear'])
net.train(train_data, epochs=100, batch_size=20,
        eval_data=eval_data,
        collect_train_stat=True,
        collect_eval_stat=True)
net.save('model_advertising.json')
