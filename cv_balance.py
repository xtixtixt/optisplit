from copy import deepcopy
import time
from datetime import timedelta
from itertools import count
import joblib
import numpy as np
import scipy.sparse as sp
from pdb import set_trace as bp
from functools import reduce
from itertools import permutations
import pandas as pd

def pos_neg_ratio(targets):
    pos = np.array(targets.sum(axis=0)).ravel() # positive distributions
    neg = targets.shape[0] - pos # negative distributions
    neg[neg == 0] = 1 # avoid division by zero
    return pos / neg

def weighted_pos_neg_ratio(targets):
    t = np.array(targets.sum(axis=0)).ravel()
    t[t == 0] = targets.shape[0] # avoid division by zero
    weights = abs((targets.shape[0]-t)/t)
    return weights

def wld(folds, targets):
    tt = deepcopy(targets)
    all_d = weighted_pos_neg_ratio(tt)
    res = []
    for f in folds:
        all_f = weighted_pos_neg_ratio(tt[f[1]])
        t = np.array(tt[f[1]].mean(axis=0)).ravel()
        sums = np.array(tt[f[1]].sum(axis=0) == 0).ravel()
        t[sums] = 1/tt[f[1]].shape[0]
        res.append(abs(all_f - all_d)*t)
    res = np.stack(res)
    return res.mean(axis=0)


def cv_evaluate(folds, targets, class_sizes, method='original', evaluate_min=False):
    """Return X, Y evaluation metrics for a cv"""

    if evaluate_min:
        targets2 = deepcopy(targets)
        pos_index = np.where(class_sizes > 0.5*targets.shape[0])[0]
        targets[:,pos_index] = (targets[:,pos_index] == 0).astype(np.int)
        class_sizes = targets.sum(axis=0)

    test_sets = [f[1] for f in folds]
    target_sums = [targets[t].sum(axis=0) for t in test_sets]
    max_subset_size = np.stack([np.array(t) for t in target_sums])[:,0,:].max(axis=0)

    min_subset_size = np.stack([np.array(t) for t in target_sums])[:,0,:].min(axis=0)

    x0 = class_sizes - max_subset_size
    x1 = max_subset_size / class_sizes
    x3 = x0/class_sizes

    K = len(folds)
    if method == 'dcp':
        res =  (K-1)/K - x3
    elif method == 'wld':
        res = np.array(wld(folds, targets)).ravel()
    else:
        raise NotImplementedError('invalid method')

    if evaluate_min:
        targets = targets2
        res = x0
        return np.array(res).ravel(), class_sizes
    return np.array(res).ravel()


def transfer_sequences(class_index, arr0, arr1, n_transfer, A, targets, sequences=None):
    """Transfer contents of class_index array from arr0 to arr1"""
    arr0_index = np.intersect1d(class_index, arr0).astype(np.int)
    # select sequences with smallest number of other features
    tt = np.array(targets[arr0_index, :].sum(axis=1)).ravel()

    if sequences is not None:
        # use precomputed transfer index
        transfer_index = sequences
    else:
        # select sequences with fewest other classes to be transferred
        transfer_index = arr0_index[tt.argsort()[:n_transfer]]

    # move arr0 to arr1
    arr1 = np.concatenate((arr1, transfer_index)).astype(np.int)
    arr0 = np.setdiff1d(arr0, transfer_index).astype(np.int)
    return arr0, arr1, transfer_index

def balance(targets, A, folds, n_splits):

    n_transfer = calc_transfer(targets, A, folds, n_splits)
    class_index = np.where(targets[:,A].toarray().ravel() > 0)[0]
    excess = np.array([])
    # process folds with too many test cases
    for i, n in enumerate(n_transfer):
        if n_transfer[i] < 0:
            tr_index = folds[i][0]
            test_index = folds[i][1]
            test_index, tr_index, transfer_index = transfer_sequences(class_index, test_index, tr_index, abs(n_transfer[i]), A, targets)
            excess = np.concatenate((excess, transfer_index))
            folds[i] = tr_index, test_index #?
        else:
            continue

    # process folds with too few test cases
    for i, n in enumerate(n_transfer):
        if n_transfer[i] > 0:
            tr_index = folds[i][0]
            test_index = folds[i][1]
            sequences = excess[:abs(n_transfer[i])]
            excess = np.setdiff1d(excess, sequences)
            tr_index, test_index, transfer_index = transfer_sequences(class_index, tr_index, test_index, n_transfer[i], A, targets, sequences=sequences)
            folds[i] = tr_index, test_index #?
        else:
            continue

    assert len(excess) == 0,'Failed to distribute all sequences'

    return folds, n_transfer

def check_folds(folds, targets):
    all_sequences_in_test = sum([len(np.unique(f[1])) for f in folds]) == targets.shape[0]
    separate_training_test = all([len(np.intersect1d(f[0], f[1])) == 0 for f in folds])
    data_shape = all([len(f[0]) + len(f[1]) == targets.shape[0] for f in folds])
    no_overlapping_test_sets = len(np.unique(np.concatenate([np.unique(f[1]) for f in folds]))) == len(np.concatenate([f[1] for f in folds]))
    return all_sequences_in_test and no_overlapping_test_sets and separate_training_test and data_shape

def random_cv(n_splits, targets, seed=42):
    np.random.seed(seed)
    t = np.arange(targets.shape[0])
    np.random.shuffle(t)
    folds = np.array_split(t, n_splits)
    folds = [(np.setdiff1d(t,f), f) for f in folds]
    return folds

def calc_transfer(targets, A, folds, n_splits):

    # calculate the amount of balancing needed
    tt = np.array([targets[f[1], A].sum() for f in folds])
    n_transfer = np.array([tt.sum()//n_splits - t for t in tt])
    if sum(n_transfer) < 0:
        aa = np.zeros(len(n_transfer)).astype(np.int)
        aa[:abs(sum(n_transfer))] = 1
        n_transfer = n_transfer + aa

    assert sum(n_transfer) == 0, 'Balancing failed'
    return n_transfer

def optisplit(n_splits, targets, method='wld', max_epochs=3, seed=42, initial_folds=None):

    np.random.seed(seed)

    targets = sp.csr_matrix(targets)
    class_sizes = targets.sum(axis=0)

    # if > 50% of the examples are positive, optimize the negative distribution
    pos_index = np.where(class_sizes > 0.5*targets.shape[0])[0]
    targets[:,pos_index] = (targets[:,pos_index] == 0).astype(np.int)
    class_sizes = targets.sum(axis=0)

    if initial_folds is None:
        folds0 = random_cv(n_splits, targets)
    else:
        folds0 = initial_folds
    res0 = cv_evaluate(folds0, targets, class_sizes, method=method)

    score0 = np.sum(res0)

    start = time.time()

    for jjj in range(max_epochs):
        max_offset = 0
        print(f'round {jjj}')
        if jjj == 0:
            print(score0)

        for iii in range(targets.shape[1]):
            folds = deepcopy(folds0)

            A = np.argsort(np.array(res0).ravel())[::-1][max_offset]

            folds, n_transfer = balance(targets, A, folds, n_splits)

            res1 = cv_evaluate(folds, targets, class_sizes, method=method)

            if np.sum(res0) <= np.sum(res1) or np.all(n_transfer == 0):
                #balancing unbalanced some other classes
                max_offset += 1
                continue

            score1 = np.sum(res1)
            folds0 = folds
            res0 = res1
        print(score1)
        if  np.isclose(score0, score1, atol=0.1):
            break

    assert check_folds(folds, targets), 'Invalid CV folds created'

    print(f'Time: {str(timedelta(seconds=time.time()-start))}')
    print(f'Ignored {max_offset} classes')
    return folds0

def main():
    pass

if __name__ == '__main__':
    main()


