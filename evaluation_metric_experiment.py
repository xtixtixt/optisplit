import numpy as np
import warnings
from pdb import set_trace as bp
import joblib
import scipy.sparse as sp
from copy import deepcopy
import matplotlib.pyplot as plt


from cv_comparison_experiment import ld
import cv_balance

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
warnings.filterwarnings('ignore', message='Comparing a sparse matrix with 0 using == is inefficient')


def equal(y, ones, n_folds):
    """Equally distributed folds"""

    for j, yy in enumerate(y):
        for i in range(yy.shape[1]):
            yy[:ones[i]//n_folds, i] = 1
    targets = np.row_stack(y)
    return sp.csr_matrix(targets)

def classes_missing_from_1_fold(y,  ones, n_folds):
    for j, yy in enumerate(y):
        if j == 0:
            continue
        else:
            for i in range(yy.shape[1]):
                yy[:ones[i]//(n_folds-1), i] = 1
    targets = np.row_stack(y).astype(np.int)
    return sp.csr_matrix(targets)

def difference(y,  ones, n_folds):
    """Small difference between folds"""
    diff = 0.2
    for j, yy in enumerate(y):
        if j == 0:
            for i in range(yy.shape[1]):
                yy[:ones[i]//n_folds+(diff*(ones[i]//n_folds)).astype(np.int), i] = 1
        elif j== 1:
            for i in range(yy.shape[1]):
                yy[:ones[i]//n_folds-(diff*(ones[i]//n_folds)).astype(np.int), i] = 1
        else:
            for i in range(yy.shape[1]):
                yy[:ones[i]//n_folds, i] = 1
    # bp() NOTE negatives are not distributed the same way
    # (y[1] == 0).astype(np.int).sum(axis=0) / (y[0] == 0).astype(np.int).sum(axis=0)

    targets = np.row_stack(y)
    return sp.csr_matrix(targets)



def mk_y(size, n_folds):
    y = np.split(np.zeros(size), n_folds)
    folds = np.split(np.arange(size[0]), n_folds)
    folds = [(np.setdiff1d(np.arange(size[0]), f), f) for f in folds]

    ones = np.linspace(start=2*n_folds, stop=size[0]//2, num=100).astype(np.int)

    res = {}
    res['Equal'] = folds, equal(deepcopy(y),  ones, n_folds)
    res['Difference'] = folds, difference(deepcopy(y),  ones, n_folds)
    res['One missing'] = folds, classes_missing_from_1_fold(deepcopy(y),  ones, n_folds)

    joblib.dump(res, 'results/res.joblib')


def plot_metrics():

    datas  = joblib.load('results/res.joblib')
    methods = ['rld', 'ld', 'dcp']

    for method in methods:
        plt.clf()
        for name in datas:
            data = datas[name]
            neg_targets = (data[1] == 0).astype(np.int)
            if method=='rld':
                res= np.array(cv_balance.rld(data[0], data[1])).ravel()
            elif method=='ld':
                res = ld(data[0], data[1], np.arange(data[1].shape[1]))
            elif method=='dcp':
                res= cv_balance.cv_evaluate(data[0], data[1], np.array(data[1].sum(axis=0)).ravel(), method=method)
            else:
                raise NotImplementedError

            sizes = np.array(data[1].sum(axis=0)).ravel()
            markers = {'Equal':'.', 'One missing':'+', 'Difference':'2'}
            plt.semilogx(sizes, res, '.', ms=11, label=name, color='k', marker=markers[name], markevery=0.04 )


        if method == 'dcp':
            method = 'DCP'
        if method == 'rld':
            method = 'rLD'
        if method == 'ld':
            method = 'LD'

        params = {'legend.fontsize': '16',
                  'figure.figsize': (4.4, 3.3),
                 'axes.labelsize': '16',
                 'axes.titlesize':'16',
                 'xtick.labelsize':'16',
                 'ytick.labelsize':'16',
                 'font.family':'serif'}

        plt.rcParams.update(params)
        plt.locator_params(axis='y', nbins=4)

        plt.xlabel('Class size')
        plt.ylabel('Class score')
        if method == 'LD':
            plt.legend()
        plt.savefig(f'results/{method}.pdf')

if __name__ == '__main__':
    mk_y((100000,100), 10)
    plot_metrics()
