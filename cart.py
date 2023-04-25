#!/usr/bin/env python3
import itertools
import logging
import numpy as np
import log_setup

class TreeNode(object):
    def __init__(self, feat_name, feat_idx, feat_val, samp_cnt, left=None, right=None):
        self._feat_name = feat_name
        self._feat_idx = feat_idx
        self._feat_val = feat_val
        self._samp_cnt = samp_cnt

        self._left     = left
        self._right    = right

    def to_string(self):
        return [
                'feat_name:{}, feat_idx:{}, feat_val:{}, samp_cnt:{}'.format(
                    self._feat_name,
                    self._feat_idx, self._feat_val, self._samp_cnt),
                self._left.to_string(),
                self._right.to_string(),
                ]


class LeafNode(object):
    def __init__(self, value, response_values, left=None, right=None):
        self._value = value
        self._samp_cnt  = len(response_values)
        self._uniq, self._cnts = np.unique(response_values, return_counts=True)
        self._left  = left
        self._right = right

    def to_string(self):
        return ['value:{}, samp_cnt:{}, uniq:{}, cnts:{}'.format(self._value, self._samp_cnt, self._uniq, self._cnts)]


class CartClassification(object):
    def __init__(self):
        self._min_split_info_gain = 1e-6
        self._min_split_size      = 3

    def train(self, names, samples, cate_feats):
        self._feat_names = names
        self._samples    = samples
        # key:   feature's column index in samples
        # value: string value to numeric value
        self._cate_feats = cate_feats
        self._cart_tree  = self._recursive_tree(np.arange(len(self._samples)))

    def _binary_split(self, samples, indices, feat_idx, feat_val):
        if feat_idx not in self._cate_feats:
            # numerical feature, feat_val is a float
            pos = samples[indices, feat_idx] <= feat_val
        else:
            # categorical feature, feat_val is a list
            pos = np.isin(samples[indices, feat_idx], feat_val)
        l_indices = indices[pos]
        r_indices = indices[np.logical_not(pos)]
        return l_indices, r_indices

    def _shannon_entropy(self, unique, counts):
        prob = 1.0 * counts / np.sum(counts)
        return np.sum(- prob * np.log2(prob))

    def _calculate_split_entropy(self, samples, l_indices, r_indices):
        l_entropy = self._shannon_entropy(*np.unique(samples[l_indices,-1], return_counts=True))
        r_entropy = self._shannon_entropy(*np.unique(samples[r_indices,-1], return_counts=True))
        l_count, r_count = len(l_indices), len(r_indices)
        return 1.0 * (l_entropy * l_count + r_entropy * r_count) / (l_count + r_count)


    def _find_best_split(self, indices):
        unique, counts = np.unique(self._samples[indices, -1], return_counts=True)
        if len(unique) == 1:
            return None, unique[0]
        if len(indices) < self._min_split_size: # most frequent
            return None, unique[np.argmax(counts)]

        base_entropy = self._shannon_entropy(unique, counts)

        best_feat_idx  = -1
        best_split_val = None # either numerical, or a list
        best_info_gain = 0.0
        sz = len(indices)
        for feat_idx in range(self._samples.shape[1] - 1):
            if feat_idx not in self._cate_feats:
                # numerical feature
                for split_value in np.unique(self._samples[indices, feat_idx]):
                    l_indices, r_indices = self._binary_split(self._samples, indices, feat_idx, split_value)
                    info_gain = base_entropy - self._calculate_split_entropy(self._samples, l_indices, r_indices)
                    if info_gain > best_info_gain:
                        #logging.info('{} feat #{} {} using {}: ig {:.6f} -> {:.6f}'.format(sz,
                        #    feat_idx, self._feat_names[feat_idx], split_value,
                        #    best_info_gain, info_gain))
                        best_feat_idx  = feat_idx
                        best_split_val = split_value
                        best_info_gain = info_gain
                continue

            # categorical feature
            cate_values = list(self._cate_feats[feat_idx].values())

            for a_cate in cate_values: # one-vs-(n-1)
                l_indices, r_indices = self._binary_split(self._samples, indices, feat_idx, [a_cate])
                if len(l_indices) == 0 or len(r_indices) == 0:
                    continue
                info_gain = base_entropy - self._calculate_split_entropy(self._samples, l_indices, r_indices)
                if info_gain > best_info_gain:
                    #logging.info('{} feat #{} {} using {}: ig {:.6f} -> {:.6f}'.format(sz,
                    #    feat_idx, self._feat_names[feat_idx], [a_cate],
                    #    best_info_gain, info_gain))
                    best_feat_idx  = feat_idx
                    best_split_val = [a_cate] # list
                    best_info_gain = info_gain

            if len(cate_values) > 5:
                continue

            for l in range(2, len(cate_values) // 2 + 1): # 1-4, 2-3, unnecessary 3-2 & 4-1
                for comb in itertools.combinations(cate_values, l):
                    comb = list(comb)
                    #logging.info('checking feat_idx {} comb {}'.format(feat_idx, comb))
                    l_indices, r_indices = self._binary_split(self._samples, indices, feat_idx, comb)
                    if len(l_indices) == 0 or len(r_indices) == 0:
                        continue
                    info_gain = base_entropy - self._calculate_split_entropy(self._samples, l_indices, r_indices)
                    if info_gain > best_info_gain:
                        #logging.info('{} feat #{} {} using {}: ig {:.6f} -> {:.6f}'.format(sz,
                        #    feat_idx, self._feat_names[feat_idx], comb,
                        #    best_info_gain, info_gain))
                        best_feat_idx  = feat_idx
                        best_split_val = comb
                        best_info_gain = info_gain

        if best_feat_idx < 0 or best_info_gain < self._min_split_info_gain:
            return None, unique[np.argmax(counts)] # most frequent
        else:
            return best_feat_idx, best_split_val

    def _recursive_tree(self, indices):
        feat, value = self._find_best_split(indices)
        if feat is None:
            return LeafNode(value, self._samples[indices,-1]) # value is model's response

        print('split at feat {} {} using {}'.format(feat, self._feat_names[feat], value))
        l_indices, r_indices = self._binary_split(self._samples,
                indices, feat, value)
        result = TreeNode(
                self._feat_names[feat],
                feat, value, len(indices),
                left=self._recursive_tree(l_indices),
                right=self._recursive_tree(r_indices))
        result._left._parent  = result
        result._right._parent = result
        return result

if __name__ == '__main__':
    import ds_heart
    import pprint
    log_setup.setup()
    feat_names, samples, idx2map = ds_heart.load_dataset()
    cart_classification = CartClassification()
    cart_classification.train(feat_names, samples, idx2map)
    pprint.pprint(cart_classification._cart_tree.to_string(), indent=4, width=100)
