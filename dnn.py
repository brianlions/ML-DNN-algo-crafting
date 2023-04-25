import json
import logging
import time
import numpy as np

class StopWatch(object):
    def __init__(self):
        self._begin = time.time()
    def elapsed(self):
        return time.time() - self._begin

def activation_builder(name):
#{
    class Linear(object):
        def activate(self, z):
            return z
        def derivative(self, z):
            return np.ones(z.shape)
        def name(self):
            return 'linear'
    class Sigmoid(object):
        def activate(self, z):
            return 1.0 / (1.0 + np.exp(-z))
        def derivative(self, z):
            act = self.activate(z)
            return act * (1.0 - act)
        def name(self):
            return 'sigmoid'
    class Tanh(object):
        def activate(self, z):
            pos = np.exp(z)
            neg = np.exp(-z)
            return (pos - neg) / (pos + neg)
        def derivative(self, z):
            act = self.activate(z)
            return (1.0 - act) * (1.0 + act)
        def name(self):
            return 'tanh'
    class Relu(object):
        def activate(self, z):
            return np.clip(z, a_min=0.0, a_max=None)
        def derivative(self, z):
            return np.array(z >= 0, dtype=float)
        def name(self):
            return 'relu'

    if name == 'linear':
        return Linear()
    elif name == 'sigmoid':
        return Sigmoid()
    elif name == 'tanh':
        return Tanh()
    elif name == 'relu':
        return Relu()
    else:
        raise Exception('invalid activation: {}'.format(name))
#}

def loss_builder(name, act_name=None):
#{
    class MeanSquaredError(object):
        def __init__(self, act_name):
            self._activation = activation_builder(act_name)
        def loss(self, a, y):
            return 0.5 * np.linalg.norm(a - y) ** 2
        def delta(self, z, a, y):
            return (a - y) * self._activation.derivative(z)
        def name(self):
            return 'MeanSquaredError'
    class CrossEntropyLoss(object):
        def loss(self, a, y):
            return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)), axis=-2)
        def delta(self, z, a, y):
            return a - y
        def name(self):
            return 'CrossEntropy'

    if name in ('mse', 'MeanSquaredError'):
        return MeanSquaredError(act_name)
    elif name in ('ce', 'CrossEntropy'):
        return CrossEntropyLoss()
    else:
        raise Exception('invalid loss: {}'.format(name))
#}

class NeuralNetwork(object):
    def __init__(self, input_size, fc_sizes, lr, loss,
            l2reg=0.0, l1reg=0.0, dropout=0.0, momentum=0.0,
            activations=[],
            initializer='xavier', # simple, xavier
            stride=8):
    #{
        if activations and len(activations) != len(fc_sizes):
            raise Exception('check failed len(activations) != len(fc_sizes): {} vs. {}'.format(
                len(activations), len(fc_sizes)))
        if dropout > 0 and momentum > 0:
            raise Exception('both dropout={:.g} and momentum={:.g} are enabled'.format(
                dropout, momentum))
        if not activations:
            activations = ['sigmoid' for _ in range(len(sizes))]

        self._input_size = input_size
        self._fc_sizes   = fc_sizes
        self._lr         = lr
        self._loss       = loss_builder(loss, activations[-1])

        self._initializer = initializer
        self._weights     = []
        self._biases      = []
        self._act_fns     = []
        for lyr in range(len(self._fc_sizes)):
            low_size  = lyr == 0 and self._input_size or self._fc_sizes[lyr-1]
            high_size = self._fc_sizes[lyr]
            if self._initializer == 'xavier':
                sigma = np.sqrt(1.0 / low_size)
            elif self._initializer == 'simple':
                sigma = 1.0
            else:
                raise Exception('unknown self._initializer: {}'.format(self._initializer))
            self._weights.append(np.random.normal(0, sigma, size=(low_size, high_size)))
            self._biases.append(np.random.normal(0, 1.0, size=high_size))
            self._act_fns.append(activation_builder(activations[lyr]))
            logging.info('init fc-layer #{}: w shape {} mean {:.6f}, b shape {} mean {:.6f}, act {}'.format(
                lyr + 1,
                self._weights[lyr].shape, np.mean(self._weights[lyr]),
                self._biases[lyr].shape,  np.mean(self._biases[lyr]),
                self._act_fns[lyr].name()))

        self._l2reg, self._l1reg, self._dropout = l2reg, l1reg, dropout
        self._momentum = momentum
        self._stride = stride

        # dropout masks
        self._dropout_masks = [np.ones(w.shape)  for w in self._weights]
        if self._momentum > 0:
            self._velocities = [np.zeros(w.shape) for w in self._weights]
        else:
            self._velocities = None
        self._batch_nabla_w = [np.zeros(w.shape) for w in self._weights]
        self._batch_nabla_b = [np.zeros(b.shape) for b in self._biases]
    #}

    def _expand_vec2mat(self, y):
    #{
        if self._fc_sizes[-1] == 1:
            # while fitting regression models, if y shape is (n,) instead of (n,1),
            # in function MSE.loss(a, y), both a and y will be broadcasted, the result is not expected
            # e.g.
            #   print(np.array([1,2,3]).reshape(3,1) - np.array([10,20,30]))
            #     [[-9 -19 -29]
            #      [-8 -18 -28]
            #      [-7 -17 -27]]
            if y.ndim == 1:
                # row vector -> column vector
                return np.expand_dims(y, axis=1)
            else:
                return y
        matrix = np.zeros((len(y), self._fc_sizes[-1]), dtype=float)
        for idx, v in enumerate(y):
            matrix[idx][v] = 1
        return matrix
    #}

    def train(self, train_data, epochs, batch_size,
            eval_data=None,
            collect_eval_stat=False,
            collect_train_stat=False):
    #{
        train_x, train_y = train_data[0], self._expand_vec2mat(train_data[1])
        n_samples = len(train_data[0])
        if eval_data:
            n_eval = len(eval_data[0])
        if batch_size <= 0:
            batch_size = n_samples

        self._info = ', '.join([
                'lr={:g}'.format(self._lr),
                'l2reg={:g}'.format(self._l2reg),
                'l1reg={:g}'.format(self._l1reg),
                'dropout={:g}'.format(self._dropout),
                'momentum={:g}'.format(self._momentum),
                'stride={}'.format(self._stride),
                'epochs={}'.format(epochs),
                'batch_size={}'.format(batch_size),
                'n_samples={}'.format(n_samples),
                ])
        logging.info('training args: {}'.format(self._info))

        train_stat, eval_stat = [], []
        for ep in range(epochs):
            # shuffle training samples
            shuffle_idx = np.random.permutation(n_samples)
            np.take(train_x, shuffle_idx, axis=0, out=train_x)
            np.take(train_y, shuffle_idx, axis=0, out=train_y)

            # batch training
            sw = StopWatch()
            for pos in range(0, n_samples, batch_size):
                batch_x = train_x[pos : pos + batch_size]
                batch_y = train_y[pos : pos + batch_size]
                self.batch_train(batch_x, batch_y)
            logging.info('{:.3f} sec, finish epoch {:d} / {:d}'.format(sw.elapsed(), ep + 1, epochs))

            # validation
            if collect_train_stat:
                loss      = self.__total_loss(train_x, train_y, label_is_matrix=True)
                pred_lbl  = self.evaluate(train_x,     train_y, label_is_matrix=True)
                n_correct = np.sum(pred_lbl[:,0] == pred_lbl[:,1])
                train_stat.append([n_correct, loss[0], loss[1]]) # correct, avg lost, reg loss
                logging.info('dataset train: loss {:.4f} {:.4f}, accuracy {:5d} / {:5d} = {:.4f}'.format(
                    loss[0], loss[1], n_correct, n_samples, 1.0 * n_correct / n_samples))
            if collect_eval_stat and eval_data is not None:
                loss      = self.__total_loss(eval_data[0], eval_data[1], label_is_matrix=False)
                pred_lbl  = self.evaluate(eval_data[0],     eval_data[1], label_is_matrix=False)
                n_correct = np.sum(pred_lbl[:,0] == pred_lbl[:,1])
                eval_stat.append([n_correct, loss[0], loss[1]]) # correct, avg lost, reg loss
                logging.info('dataset eval:  loss {:.4f} {:.4f}, accuracy {:5d} / {:5d} = {:.4f}'.format(
                    loss[0], loss[1], n_correct, n_eval, 1.0 * n_correct / n_eval))

        return np.array(train_stat), np.array(eval_stat)
    #}

    def predict(self, x):
        for w, b, af in zip(self._weights, self._biases, self._act_fns):
            x = af.activate(np.matmul(x, w) + b)
        return x

    def evaluate(self, all_x, all_y, label_is_matrix=False):
        '''
        Returns:
            numpy array of [prediction, label]
        '''
        if label_is_matrix:
            all_y = np.argmax(all_y, axis=1)
        return np.stack([np.argmax(self.predict(all_x), axis=1), all_y], axis=1)

    def save(self, filename, round_decimals=8):
    #{
        logging.info('saving model to file: {}'.format(filename))
        if round_decimals > 0:
            weights = [np.round(w, decimals=round_decimals).tolist() for w in self._weights]
            biases  = [np.round(b, decimals=round_decimals).tolist() for b in self._biases]
            if self._momentum > 0:
                velocities = [np.round(v, decimals=round_decimals).tolist() for v in self._velocities]
        else:
            weights = [w.tolist() for w in self._weights]
            biases  = [b.tolist() for b in self._biases]
            if self._momentum > 0:
                velocities = [v.tolist() for v in self._velocities]

        with open(filename, 'w') as fh:
            data_obj = {
                'info':        self._info,
                'input_size':  self._input_size,
                'fc_sizes':    self._fc_sizes,
                'lr':          self._lr,
                'loss':        self._loss.name(),
                'l2reg':       self._l2reg,
                'l1reg':       self._l1reg,
                'dropout':     self._dropout,
                'momentum':    self._momentum,
                'stride':      self._stride,
                'initializer': self._initializer,
                'activations': [a.name() for a in self._act_fns],
                'weights':     weights,
                'biases':      biases,
                }
            if self._momentum > 0:
                data_obj['velocities'] = velocities
            json.dump(data_obj, fh, indent=0)
    #}

    @staticmethod
    def load(filename):
    #{
        logging.info('loading model from file: {}'.format(filename))
        with open(filename, 'r') as fh:
            data = json.load(fh)
            net = NeuralNetwork(
                    input_size  = data['input_size'],
                    fc_sizes    = data['fc_sizes'],
                    lr          = data['lr'],
                    loss        = data['loss'],
                    l2reg       = data['l2reg'],
                    l1reg       = data['l1reg'],
                    dropout     = data['dropout'],
                    momentum    = data['momentum'],
                    stride      = data['stride'],
                    initializer = data['initializer'],
                    activations = data['activations'],
                    )
            net._info    = data['info']
            net._weights = [np.array(w) for w in data['weights']]
            net._biases  = [np.array(b) for b in data['biases']]
            if data['momentum'] > 0:
                net._velocities = [np.array(v) for v in data['velocities']]
            else:
                net._velocities = None
            return net
    #}

    def batch_train(self, batch_x, batch_y):
    #{
        # initialization
        for idx in range(len(self._batch_nabla_w)):
            self._dropout_masks[idx].fill(1.0)      # NOTE: fill with 1.0 !!!
            self._batch_nabla_w[idx].fill(0.0)
            self._batch_nabla_b[idx].fill(0.0)
        if self._dropout and len(self._fc_sizes) > 1:
            flags = [np.random.binomial(1, 1.0 - self._dropout, [sz]) for sz in self._fc_sizes[:-1]]
            for l, f in enumerate(flags):
                self._dropout_masks[l]   *= f                  # row-wise broadcast
                self._dropout_masks[l+1] *= f.reshape((-1, 1)) # col-wise broadcast

        batch_size = len(batch_x)
        # forward and backward
        for offset in range(0, batch_size, self._stride):
            delta_nabla_w, delta_nabla_b = self.__back_propagation(
                    batch_x[offset : offset + self._stride],
                    batch_y[offset : offset + self._stride])
            for idx in range(len(self._batch_nabla_w)):
                self._batch_nabla_w[idx] += np.sum(delta_nabla_w[idx], axis=0)
                self._batch_nabla_b[idx] += np.sum(delta_nabla_b[idx], axis=0)

        # gradient descent
        for idx in range(len(self._fc_sizes)):
            if self._momentum > 0:
                self._velocities[idx] *= self._momentum
                self._velocities[idx] += (1 - self._momentum) * self._batch_nabla_w[idx]
                self._weights[idx]    -= self._lr / batch_size * self._velocities[idx]
                self._biases[idx]     -= self._lr / batch_size * self._batch_nabla_b[idx]
            else:
                if self._l2reg > 0:
                    self._weights[idx] *= 1 - self._lr * self._l2reg
                elif self._l1reg > 0:
                    self._weights[idx] -= self._lr * self._l1reg * np.sign(self._weights[idx])
                # stochastic gd
                self._weights[idx] -= self._lr / batch_size * self._batch_nabla_w[idx]
                self._biases[idx]  -= self._lr / batch_size * self._batch_nabla_b[idx]
    #}

    def __back_propagation(self, stride_x, stride_y):
    #{
        # forward
        act   = stride_x    # samples
        all_a = [stride_x]  # samples
        all_z = []
        for m, w, b, af in zip(self._dropout_masks, self._weights, self._biases, self._act_fns):
            if self._dropout and len(self._fc_sizes) > 1:
                z = np.matmul(act, m * w) + b
            else:
                z = np.matmul(act, w) + b
            act = af.activate(z)
            all_z.append(z)
            all_a.append(act)

        # backward
        nabla_w = [None for idx in range(len(self._weights))]
        nabla_b = [None for idx in range(len(self._biases))]
        # shapes:
        #   a[i], z[i], delta[i]:               (mini_batch_size, n_neurons[i])
        #   np.expand_dims(a[i-1],   axis=2):   (mini_batch_size, n_neurons[i-1], 1)
        #   np.expand_dims(delta[i], axis=1):   (mini_batch_size, 1, n_neurons[i])
        #   after matmul, shape of nabla_w[i]:  (mini_batch_size, n_neurons[i-1], n_neurons[i])
        delta       = self._loss.delta(all_z[-1], all_a[-1], stride_y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(np.expand_dims(all_a[-2], axis=2),
                                np.expand_dims(delta, axis=1))
        for l in range(2, len(self._fc_sizes) + 1):
            if self._dropout and len(self._fc_sizes) > 1:
                temp_matrix = self._dropout_masks[- l + 1] * self._weights[- l + 1]
            else:
                temp_matrix = self._weights[- l + 1]
            delta = self._act_fns[-l].derivative(all_z[-l]) * np.matmul(
                        delta, np.transpose(temp_matrix))
            nabla_b[-l] = delta
            nabla_w[-l] = np.matmul(np.expand_dims(all_a[- l - 1], axis=2),
                                    np.expand_dims(delta, axis=1))
        return nabla_w, nabla_b
    #}

    def __total_loss(self, all_x, all_y, label_is_matrix):
        '''
        Description:
            Computes loss on dataset all_x + all_y.  If label_is_matrix == False,
            converts verctor all_y to matrix, in case of multi-class classification.
        Returns:
            [average loss, regularized loss], if neither L2 nor L2 is used, values are equal.
        '''
    #{
        num_xs   = len(all_x)
        avg_loss = 0.0
        if not label_is_matrix:
            all_y = self._expand_vec2mat(all_y)
        for offset in range(0, num_xs, self._stride):
            xs = all_x[offset : offset + self._stride]
            ys = all_y[offset : offset + self._stride]
            avg_loss += self._loss.loss(self.predict(xs), ys)
        avg_loss   = np.sum(avg_loss) / num_xs
        final_loss = avg_loss
        if self._l2reg > 0:
            final_loss += 0.5 * self._l2reg * np.sum([np.linalg.norm(w) ** 2 for w in self._weights])
        elif self._l1reg > 0:
            final_loss += self._l1reg * np.sum([np.sum(np.abs(w)) for w in self._weights])
        return avg_loss, final_loss
    #}
