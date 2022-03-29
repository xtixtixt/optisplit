import numpy as np
import joblib
import matplotlib.pyplot as plt
import scipy.sparse as sp
import warnings

from copy import deepcopy
from pdb import set_trace as bp
from textwrap import wrap

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

    targets = sp.csr_matrix(np.row_stack(y))
    return targets


def mk_y(size, n_folds):
    """Generate the synthetic data"""
    y = np.split(np.zeros(size), n_folds)
    folds = np.split(np.arange(size[0]), n_folds)
    folds = [(np.setdiff1d(np.arange(size[0]), f), f) for f in folds]

    ones = np.linspace(start=2*n_folds, stop=size[0]//2, num=100).astype(np.int)

    res = {}
    res['Equal'] = folds, equal(deepcopy(y),  ones, n_folds)
    res['Difference'] = folds, difference(deepcopy(y),  ones, n_folds)
    res['One missing'] = folds, classes_missing_from_1_fold(deepcopy(y),  ones, n_folds)



    joblib.dump(res, 'results/res.joblib')


def calculate_scores(target_fold_ratio, actual_fold_ratio):
    """Return LD and rLD scores for the given ratios"""

    #Notation like in Section 3. 
    D = 1 # data size
    Di = np.linspace(0.01*D, 0.99*D, 100) # number of positives in each class

    Sj = D*actual_fold_ratio
    Sij = Di*target_fold_ratio

    d = Di / D
    p = Sij / Sj

    rld = np.abs((d-p)/d)
    ld = np.abs(p/(1-p) - d/(1-d))

    return ld, rld


def plot_measures():
    """Plot LD and rLD scores of folds with given error"""

    # get scores
    ratios = [(0.2, 0.25), (0.2, 0.3), (0.2, 0.4), (0.2, 0.5)][::-1]
    scores = [calculate_scores(*r) for r in ratios]
    ld_scores = [s[0] for s in scores]
    rld_scores = [s[1] for s in scores]

    # plot results

    # Score comparison
    plt.figure(figsize=(11, 3.8))
    plt.subplots_adjust(wspace=0.3, top=0.90, bottom=0.15, right=0.82, left=0.10)

    Di = np.linspace(0.01, 0.99, 100)


    plt.subplot(1,2,1,)
    plt.yscale('log')
    plt.plot(Di, np.array(ld_scores).T)
    plt.xlabel('$D_i$', fontsize=13)
    plt.title('A', fontsize=16)
    plt.ylabel('LD', fontsize=13, rotation=0, labelpad=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.subplot(1,2,2,)
    plt.plot(Di, np.array(rld_scores).T)
    plt.title('B', fontsize=16)
    plt.ylabel('rLD', fontsize=13, rotation=0, labelpad=15)
    plt.xlabel('$D_i$', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


    title = 'Ratio of positive data points in the fold'
    title = '\n'.join(wrap(title, 20))

    lg = plt.legend([r[1] for r in ratios], bbox_to_anchor=(1.03, 0.8), loc="upper left", fontsize=13, title=title)
    title = lg.get_title()
    title.set_fontsize(13)

    plt.savefig(f'results/ld_vs_rld.pdf')

    # Difference comparison

    # calculate pairwise differences between scores
    ld_differences = np.array([x - y for i,x in enumerate(ld_scores[::-1]) for j,y in enumerate(ld_scores[::-1]) if i > j]).T
    rld_differences = np.array([x - y for i,x in enumerate(rld_scores[::-1]) for j,y in enumerate(rld_scores[::-1]) if i > j]).T
    labels = np.array([f'{ratios[i][1]}-{ratios[j][1]}' for i,x in enumerate(ld_scores[::-1]) for j,y in enumerate(ld_scores[::-1]) if i > j]).T

    plt.clf()
    plt.figure(figsize=(11, 3.8))
    plt.subplots_adjust(wspace=0.3, top=0.90, bottom=0.15, right=0.82, left=0.10)
    Di = np.linspace(0.01, 0.99, 100)


    plt.subplot(1,2,1,)
    plt.yscale('log')
    plt.plot(Di, ld_differences)
    plt.xlabel('$D_i$', fontsize=13)
    plt.title('C', fontsize=16)
    plt.ylabel('$\Delta LD$', fontsize=13, rotation=0, labelpad=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.subplot(1,2,2,)
    plt.plot(Di, rld_differences)
    plt.title('D', fontsize=16)
    plt.xlabel('$D_i$', fontsize=13)
    plt.ylabel('$\Delta rLD$', fontsize=13, rotation=0, labelpad=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


    plt.legend(labels, bbox_to_anchor=(1.02, 0.8), loc="upper left", fontsize=13)
    plt.savefig(f'results/ld_vs_rld_differences.pdf')


def synthetic_data_experiment():

    datas  = joblib.load('results/res.joblib')
    methods = ['rld', 'ld', 'dcp']

    for i, name in enumerate(datas):
        plt.clf()
        data = datas[name]
        for j, method in enumerate(methods):

            if method=='rld':
                res= np.array(cv_balance.rld(data[0], data[1])).ravel()
                method = 'rLD'
            elif method=='ld':
                res = cv_balance.ld(data[0], data[1])
                method = 'LD'
            elif method=='dcp':
                res= cv_balance.cv_evaluate(data[0], data[1], np.array(data[1].sum(axis=0)).ravel(), method=method)
                method = 'DCP'
            else:
                raise NotImplementedError

            sizes = np.array(data[1].sum(axis=0)).ravel()
            markers = {'rLD':'.', 'LD':'+', 'DCP':'2'}

            if i == 2 and j == 0:
                plt.figure(figsize=(6.6, 3.8))

            elif j == 0 and name == 'Difference':
                plt.figure(figsize=(5.0, 3.8))

            plt.plot(sizes, res, '.', ms=11, label=method, color='k', marker=markers[method], markevery=0.04)
            plt.xscale('symlog', linthreshx=0.000001)
            plt.yscale('symlog', linthreshy=0.000001)
            plt.ylim(-0.000001, max(res)+3)

        plt.xlabel('Class size', fontsize=16)
        plt.ylabel('Score', fontsize=16)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        if i == 2:
            plt.legend(bbox_to_anchor=(1.05, 0.5), loc="upper left", fontsize=16)
            plt.subplots_adjust(left=0.2, right=0.66)



        if name == 'Difference':
            plt.subplots_adjust(left=0.2, bottom=0.2)
        else:
            plt.subplots_adjust(left=0.1, bottom=0.2)
        plt.title(name, x=0.5, y=0.89, fontsize=16)
        plt.savefig(f'results/{name}.pdf')

if __name__ == '__main__':
    mk_y((100000,100), 10)
    synthetic_data_experiment()
    plot_measures()
