import argparse
import sys
import time

import arff
import joblib
import numpy as np
import scipy.sparse as sp

from copy import deepcopy
from datetime import timedelta
from joblib import Parallel, delayed
from pdb import set_trace as bp
from skmultilearn.model_selection import IterativeStratification

from cv_balance import optisplit, random_cv, cv_evaluate, check_folds, rld

sys.path.append('stratified_sampling_for_XML/stratify_function/')
from stratify import stratified_train_test_split

import warnings
warnings.filterwarnings('ignore', message='Comparing a sparse matrix with 0 using == is inefficient')

def load_datasets(dataset_type):
    datasets = {}
    if dataset_type == 'small':
        for dataset in [('mediamill', 101), ('bibtex', 159), ('delicious', 983)]:
            print(f'loading {dataset[0]}')
            with open(f'data/{dataset[0]}.arff') as f:
                data = arff.load(f)
                data = np.array(data['data'])
                datasets[dataset[0]] = sp.csr_matrix(data[:,-dataset[1]:].astype(np.int))

    elif dataset_type == 'go':
        for dataset in ['CC', 'MF']:
            print(f'loading {dataset}')
            data =sp.load_npz(f'data/{dataset}_targets.npz')
            class_sizes = data.sum(axis=0)
            if np.any(class_sizes == data.shape[0]):
                data = data[:, np.array(class_sizes) < data.shape[0]]
            datasets[dataset] = data

    elif dataset_type == 'xml':
        for dataset in ['BP_targets.npz', 'wiki10_31k.npz']:
            print(f'loading {dataset}')
            data =sp.load_npz(f'data/{dataset}')
            class_sizes = data.sum(axis=0)
            if np.any(class_sizes == 0):
                data = data[:, (np.array(class_sizes) > 0).ravel()]
            if np.any(class_sizes == data.shape[0]):
                data = data[:, np.array(class_sizes) < data.shape[0]]
            datasets[dataset] = data
    else:
        raise NotImplementedError('unknown datasets')

    return datasets

def iterstrat(n_folds, targets, random_state=42):
    X = np.zeros((targets.shape[0], 1))
    k_fold = IterativeStratification(n_splits=n_folds, random_state=random_state).split(X,targets)
    return list(k_fold)

def szymanski(n_folds, targets, random_state=42):
    X = np.zeros((targets.shape[0], 1))
    k_fold = IterativeStratification(n_splits=n_folds, random_state=random_state, order=2).split(X,targets)
    return list(k_fold)

def stratified(n_folds, targets, random_state=42):
    res = []

    remaining = np.arange(targets.shape[0])
    m = targets.shape[0]//n_folds
    for i in range(n_folds):

        if len(remaining) >  m and i < n_folds-1:
            s = m/len(remaining)
        else:
            s = len(remaining)

        tt = list(targets[remaining,:].tolil().rows)

        X = list(np.zeros((targets.shape[0], 1))[remaining])

        split = stratified_train_test_split(X, tt, target_test_size=s, random_state=random_state)
        remaining2 = remove(remaining, split[1])
        res.append((None, remaining[split[1]]))
        remaining = remaining2

    res = [(np.setdiff1d(np.arange(targets.shape[0]), f[1]), f[1]) for f in res]
    if not check_folds(res, targets):
        bp()
        check_folds(res, targets)
    return res


def remove(remaining, split):
    remaining2 = np.setdiff1d(remaining, remaining[split])
    return remaining2

def partitioning_cv(n_folds, targets, random_state=42):
    np.random.seed(random_state)
    frequencies = np.array(np.mean(targets, axis=0)).ravel()

    index = list(targets.tolil().rows)
    tt = [frequencies[index[i]] for i in range(len(index))]

    D = np.array([np.product(t) for t in tt])
    index = np.argsort(D)
    stratas = np.array_split(index, n_folds)
    for i in range(len(stratas)):
        np.random.shuffle(stratas[i])
    substratas = [np.array_split(s, n_folds) for s in stratas]
    folds = []
    for j in range(n_folds):
        res = []
        for i in range(n_folds):
            res.append(substratas[i][j])
        folds.append((None, np.concatenate(res).ravel()))


    folds = [(np.setdiff1d(np.arange(targets.shape[0]), f[1]), f[1]) for f in folds]
    if not check_folds(folds, targets):
        bp()
        check_folds(folds, targets)

    return folds

def improve_folds(dataset_type, random_state=42, output_dir='results'):
    np.random.seed(random_state)
    folds = joblib.load(f'{output_dir}/folds_{dataset_type}_{random_state}.joblib')

    res = {}
    for dataset in folds.keys():
        res[dataset] = {}
        for method in folds[dataset].keys():
            data = folds[dataset][method]

            folds0 = [(np.setdiff1d(np.arange(data[1].shape[0]), f[1]), f[1]) for f in data[0]]
            if not check_folds(folds0, data[1]):
                bp()
                check_folds(folds0, data[1])
            print(f'{method}')
            start = time.time()
            result = optisplit(n_splits=len(data[0]), targets=data[1], seed=random_state,initial_folds=folds0)
            elapsed = time.time()-start
            runtime = f'Time: {str(timedelta(seconds=elapsed))}'
            res[dataset][method] = result, data[1], elapsed
            print(runtime)
    joblib.dump(res, f'{output_dir}/folds_{dataset_type}_{random_state}_IMPROVED.joblib')


def create_folds(dataset_type, n_folds=5, random_state=42, output_dir='results'):

    own_dcp = lambda n_splits, targets, random_seed: optisplit(n_splits, targets, method='dcp', seed=random_seed)
    own_rld = lambda n_splits, targets, random_seed: optisplit(n_splits, targets, method='rld', seed=random_seed)

    datasets = load_datasets(dataset_type)
    if dataset_type in ['small', 'go']:
        methods = {'SS':stratified, 'PMBSRS':partitioning_cv,  'IS':iterstrat, 'SOIS':szymanski, 'own_dcp':own_dcp, 'own_rld':own_rld, 'random':random_cv}

    else:
        methods = {'own_dcp':own_dcp, 'own_rld':own_rld, 'PMBSRS':partitioning_cv, 'random':random_cv, 'SS':stratified}

    res = {}
    for dataset in datasets.keys():
        print(f'{dataset}')
        res[dataset] = {}
        for method in methods.keys():
            print(f'{method}')
            start = time.time()
            targets = datasets[dataset]
            try:
                result = methods[method](n_folds, deepcopy(targets), random_state)
                elapsed = time.time()-start
                runtime = f'Time: {str(timedelta(seconds=elapsed))}'
                res[dataset][method] = result, targets, elapsed
                print(runtime)
            except:
                print(f'Error in {method} on {dataset} - skipped')
    joblib.dump(res, f'{output_dir}/folds_{dataset_type}_{random_state}.joblib')


def ld(folds, targets, index):
    res = np.zeros(len(index))
    for i in index:
        k = len(folds)
        for j in range(k):
            index = folds[j][1]
            Sj = len(index)
            Sji = targets[index, i].getnnz()

            if Sj == Sji:
                # all examples positive in this fold
                continue

            try:
                res[i] += 1/k * np.abs(Sji/(Sj-Sji) - targets[:, i].sum()/(targets.shape[0] - targets[:, i].sum()))
            except:
                bp()
    return res

def label_distribution(folds, targets):
    """parallelised ld for large datasets"""
    n_jobs = 1
    L = targets.shape[1]
    index = np.array_split(np.arange(L), n_jobs)
    parallel = Parallel(n_jobs=n_jobs)
    scores = np.concatenate(parallel(delayed(ld)(folds, targets, index[i]) for i in range(n_jobs)))
    return scores.mean()

def example_distribution(folds, targets):
    k = len(folds)
    res = 0
    for j in range(k):
        Sj = len(folds[j][1])
        cj = targets.shape[0]*(1/k)
        res += np.abs(Sj - cj)
    return (1/k)*res

def evaluate_folds(dataset_type, random_state, output_dir):
    folds = joblib.load(f'{output_dir}/folds_{dataset_type}_{random_state}.joblib')
    res = {}
    for dataset in folds.keys():
        res[dataset] = {}
        for method in folds[dataset].keys():

            data = folds[dataset][method]

            targets = data[1]
            class_sizes = np.array(targets.sum(axis=0)).ravel()

            # remove empty classes if they exists
            targets = targets[:, np.where(class_sizes > 0)[0]]
            class_sizes = np.array(targets.sum(axis=0)).ravel()

            dcp = cv_evaluate(data[0], targets, class_sizes, method='dcp')

            ED = example_distribution(data[0], targets)
            LD = label_distribution(data[0], targets)
            rld_score = np.mean(rld(data[0], targets))
            dcp_score = np.mean(dcp)
            runtime = data[2]

            res[dataset][method] = {'ED':ED, 'LD':LD, 'dcp':dcp_score, 'rld':rld_score, 'runtime':runtime}

    tostr = lambda x: str(x).replace('[','').replace(']','').replace('\'', '')

    with open(f'{output_dir}/scores_{dataset_type}_{random_state}.csv', 'w') as f:
        fields = 'dataset, method, ED, LD, dcp, rld, runtime\n'
        f.write(fields)
        for dataset, results in res.items():
            for method, scores in results.items():
                score_str = tostr([v for v in list(scores.values())])
                f.write(f'{dataset},{method},{score_str}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_type', type=str, help='small, go or xml')
    parser.add_argument('random_state', type=int)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-e', '--evaluation', action='store_true', help='run evaluations')
    parser.add_argument('-i', '--improve', action='store_true', help='improve existing folds')
    parser.add_argument('-c', '--create', action='store_true', help='create folds')
    args = parser.parse_args()

    if args.create:
        create_folds(dataset_type=args.dataset_type, random_state=args.random_state, output_dir=args.output_dir)
    if args.evaluation:
        evaluate_folds(dataset_type=args.dataset_type, random_state=args.random_state, output_dir=args.output_dir)
    if args.improve:
        improve_folds(dataset_type=args.dataset_type, random_state=args.random_state, output_dir=args.output_dir)

