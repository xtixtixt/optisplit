import time

import numpy as np
import scipy.sparse as sp

from copy import deepcopy
from datetime import timedelta
from pdb import set_trace as bp

def rld(folds, targets):
    tt = deepcopy(targets)
    res = []
    di = np.array(tt.sum(axis=0)).ravel() / tt.shape[0]
    for f in folds:
        pij = np.array(tt[f[1]].sum(axis=0)).ravel() / len(f[1])
        res.append((abs((di - pij)/di)))
    res = np.stack(res)
    return res.mean(axis=0)

def dcp(folds, targets):
    tt = deepcopy(targets)
    res = []
    Si = np.array(tt.sum(axis=0)).ravel()
    for f in folds:
        Sji = np.array(tt[f[1]].sum(axis=0)).ravel()
        res.append(Sji)
    res = np.stack(res)
    return (res / Si).max(axis=0) - 1/len(folds)

def ld(folds, targets):
    tt = deepcopy(targets)
    res = []
    di = np.array(tt.sum(axis=0)).ravel() / tt.shape[0]
    di = np.where(di == 1, (tt.shape[0]-1)/tt.shape[0], di) # avoid division by zero
    for f in folds:
        pij = np.array(tt[f[1]].sum(axis=0)).ravel() / len(f[1])
        pij = np.where(pij == 1, (len(f[1])-1)/len(f[1]), pij)
        res.append(abs((pij/(1-pij) - di/(1-di))))
    res = np.stack(res)
    return res.mean(axis=0)

def cv_evaluate(folds, targets, method='original'):
    """Return X, Y evaluation metrics for a cv"""

    if method == 'dcp':
        res = np.array(dcp(folds, targets)).ravel()
    elif method == 'rld':
        res = np.array(rld(folds, targets)).ravel()
    elif method == 'ld':
        res = np.array(ld(folds, targets)).ravel()
    else:
        raise NotImplementedError('invalid method')

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

def optisplit(n_splits, targets, method='rld', max_epochs=3, seed=42, initial_folds=None):
    """Run Optisplit.

    Parameters
    ----------
    n_splits : int
        Number of cross validation folds

    targets : scipy csr matrix
        Target matrix

    method : str (rld or dcp), default=rld
        Optimisation method

    max_epochs: int, defauld=3
        Number of times to run optisplit over the data

    seed: int, default=42
        Random seed

    initial_folds: list, default=None
        List of numpy arrays containing cross validation fold indices. These
        are used as the initial folds.

    Returns
    -------
    list
        list of n_split tuples containing numpy arrays containing training and test fold indices.
    """


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
    res0 = cv_evaluate(folds0, targets, method=method)

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

            res1 = cv_evaluate(folds, targets, method=method)

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


