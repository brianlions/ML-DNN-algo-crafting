#!/usr/bin/env python3
import numpy as np

def load_dataset(filename):
    data = []
    with open(filename, 'r') as fh:
        for line in fh:
            values = list(map(float, line.strip().split('\t')))
            data.append(values)
    return np.array(data, dtype=float)

def bin_split_dataset(dataset, feat_idx, feat_val):
    pos   = dataset[:,feat_idx] <= feat_val
    left  = dataset[pos]
    right = dataset[np.logical_not(pos)]
    return left, right

def reg_leaf(dataset):
    return np.mean(dataset[:,-1])

def reg_error(dataset):
    return np.var(dataset[:,-1]) * len(dataset)

def choose_best_split(dataset, leaf_type, error_type,
        error_tolerance=1, subset_size=4):
    if len(set(dataset[:,-1])) == 1:
        return None, leaf_type(dataset)
    n_samples, n_cols = dataset.shape
    base_error = error_type(dataset)
    best_error = np.inf
    best_feat  = -1
    best_value = 0
    for feat_idx in range(n_cols - 1):
        for split_value in set(dataset[:,feat_idx]):
            l_data, r_data = bin_split_dataset(dataset, feat_idx, split_value)
            if len(l_data) < subset_size or len(r_data) < subset_size:
                continue
            temp_error = error_type(l_data) + error_type(r_data)
            if temp_error < best_error:
                best_error = temp_error
                best_feat  = feat_idx
                best_value = split_value
    if best_feat < 0 or (base_error - best_error) < error_tolerance:
        return None, leaf_type(dataset)
    return best_feat, best_value

def create_tree(dataset,
        leaf_type=reg_leaf,
        error_type=reg_error,
        error_tolerance=1, subset_size=4):
    feat, value = choose_best_split(dataset, reg_leaf, reg_error,
            error_tolerance=error_tolerance, subset_size=subset_size)

    size = len(dataset)
    if feat is None:
        return value
        #return value, size
    l_dataset, r_dataset = bin_split_dataset(dataset, feat, value)
    l_tree = create_tree(l_dataset, leaf_type, error_type,
            error_tolerance=error_tolerance, subset_size=subset_size)
    r_tree = create_tree(r_dataset, leaf_type, error_type,
            error_tolerance=error_tolerance, subset_size=subset_size)
    return {
            'count':      size,
            'feat_idx': feat,
            'feat_val': value,
            'left':       l_tree,
            'right':      r_tree,
            }

def tree_size(root):
    if isinstance(root, dict):
        l_sz = tree_size(root['left'])
        r_sz = tree_size(root['right'])
        return 1 + l_sz + r_sz
    else:
        return 1

def is_tree(node):
    return isinstance(node, dict)

def get_mean(tree):
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, test_data):
    if not len(test_data):
        return get_mean(tree)
    if is_tree(tree['left']) or is_tree(tree['right']):
        l_test, r_test = bin_split_dataset(test_data, tree['feat_idx'], tree['feat_val'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_test)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_test)
    if is_tree(tree['left']) or is_tree(tree['right']):
        # cannot merge if either child is a tree
        return tree
    else:
        # both children are leaf-nodes, decide whether they should be merged or not
        l_test, r_test = bin_split_dataset(test_data, tree['feat_idx'], tree['feat_val'])
        error_no_merge = np.linalg.norm(l_test[:,-1] - tree['left']) ** 2 \
                + np.linalg.norm(r_test[:,-1] - tree['right']) ** 2
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = np.linalg.norm(test_data[:,-1] - tree_mean) ** 2
        if error_merge < error_no_merge:
            return tree_mean
        else:
            return tree
