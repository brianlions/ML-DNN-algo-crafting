#!/usr/bin/env python3
import numpy as np

def load_dataset(filename):
    data = []
    with open(filename, 'r') as fh:
        for line in fh:
            values = list(map(float, line.strip().split('\t')))
            data.append(values)
    return np.array(data, dtype=float)

def reg_leaf(dataset):
    return np.mean(dataset[:,-1])

def reg_error(dataset):
    return np.var(dataset[:,-1]) * len(dataset)

class RegressionTree(object):
    def __init__(self):
        pass

    def _binary_split(self, dataset, indices, feat_idx, feat_val):
        pos = dataset[indices, feat_idx] <= feat_val
        l_indices = indices[pos]
        r_indices = indices[np.logical_not(pos)]
        return l_indices, r_indices

    def _find_best_split(self, indices):
        if len(set(self._dataset[indices, -1])) == 1:
            return None, self._leaf_type(self._dataset[indices])

        n_cols = self._dataset.shape[1]
        base_error = self._error_type(self._dataset[indices])
        best_error = np.inf
        best_feat  = -1
        best_value = 0
        for feat_idx in range(n_cols - 1):
            for split_value in set(self._dataset[indices, feat_idx]):
                l_indices, r_indices = self._binary_split(self._dataset,
                        indices, feat_idx, split_value)
                if len(l_indices) < self._subset_size \
                        or len(r_indices) < self._subset_size:
                    continue
                temp_error = self._error_type(self._dataset[l_indices]) \
                        + self._error_type(self._dataset[r_indices])
                if temp_error < best_error:
                    best_error = temp_error
                    best_feat  = feat_idx
                    best_value = split_value
        if best_feat < 0 or (base_error - best_error) < self._error_tolerance:
            return None, self._leaf_type(self._dataset[indices])
        return best_feat, best_value

    def train(self, dataset,
            leaf_type=reg_leaf,
            error_type=reg_error,
            error_tolerance=1,
            subset_size=4):
        self._dataset         = dataset
        self._leaf_type       = leaf_type
        self._error_type      = error_type
        self._error_tolerance = error_tolerance
        self._subset_size     = subset_size
        cart_tree = self._recursive_tree(np.arange(len(self._dataset)))
        return cart_tree

    def _recursive_tree(self, indices):
        feat, value = self._find_best_split(indices)
        size = len(indices)
        if feat is None:
            return value
        l_indices, r_indices = self._binary_split(self._dataset, indices, feat, value)
        return {
                'count':    size,
                'feat_idx': feat,
                'feat_val': value,
                'left':     self._recursive_tree(l_indices),
                'right':    self._recursive_tree(r_indices),
                }

    @staticmethod
    def size(node):
        if not isinstance(node, dict):
            return 1

        l_sz = RegressionTree.size(node['left'])
        r_sz = RegressionTree.size(node['right'])
        return 1 + l_sz + r_sz

    def __is_tree(self, node):
        return isinstance(node, dict)

    def __get_mean(self, tree):
        if self.__is_tree(tree['left']):
            tree['left'] = self.__get_mean(tree['left'])
        if self.__is_tree(tree['right']):
            tree['right'] = self.__get_mean(tree['right'])
        return (tree['left'] + tree['right']) / 2.0

    def prune(self, tree, test_data):
        return self.__recursive_prune(tree, test_data, np.arange(len(test_data)))

    def __recursive_prune(self, tree, test_data, test_indices):
        if not len(test_indices):
            return self.__get_mean(tree)

        if self.__is_tree(tree['left']) or self.__is_tree(tree['right']):
            l_indices, r_indices = self._binary_split(test_data, test_indices,
                    tree['feat_idx'], tree['feat_val'])

        if self.__is_tree(tree['left']):
            tree['left'] = self.__recursive_prune(tree['left'], test_data, l_indices)
        if self.__is_tree(tree['right']):
            tree['right'] = self.__recursive_prune(tree['right'], test_data, r_indices)

        if self.__is_tree(tree['left']) or self.__is_tree(tree['right']):
            # cannot merge if either child is a tree
            return tree
        else:
            # both children are leaf-nodes, decide whether they should be merged or not
            l_indices, r_indices = self._binary_split(test_data, test_indices,
                    tree['feat_idx'], tree['feat_val'])
            error_no_merge = np.linalg.norm(test_data[l_indices,-1] - tree['left']) ** 2 \
                    + np.linalg.norm(test_data[r_indices,-1] - tree['right']) ** 2
            tree_mean = (tree['left'] + tree['right']) / 2.0
            error_merge = np.linalg.norm(test_data[test_indices,-1] - tree_mean) ** 2
            if error_merge < error_no_merge:
                return tree_mean
            else:
                return tree

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    import pprint
    tree = RegressionTree().train(load_dataset('datasets/tree_ex00.txt'))
    print('#1 tree:')
    pprint.pprint(tree)
    print('#1 size: {}'.format(RegressionTree.size(tree)))

    tree = RegressionTree().train(load_dataset('datasets/tree_ex0.txt'))
    print('#2 tree:')
    pprint.pprint(tree)
    print('#2 size: {}'.format(RegressionTree.size(tree)))

    tree = RegressionTree().train(load_dataset('datasets/tree_ex2.txt'))
    print(RegressionTree.size(tree))
    tree = RegressionTree().train(load_dataset('datasets/tree_ex2.txt'), error_tolerance=100, subset_size=4)
    print(RegressionTree.size(tree))
    tree = RegressionTree().train(load_dataset('datasets/tree_ex2.txt'), error_tolerance=1000, subset_size=4)
    print(RegressionTree.size(tree))

    tree = RegressionTree().train(load_dataset('datasets/tree_ex2.txt'), error_tolerance=0, subset_size=1)
    sz1 = RegressionTree.size(tree)
    RegressionTree().prune(tree, load_dataset('datasets/tree_ex2test.txt'))
    sz2 = RegressionTree.size(tree)
    print('prune size: {} -> {}'.format(sz1, sz2))
