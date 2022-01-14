#!/usr/bin/env python
# coding: utf-8

from pylab import *
import time
from supervised_parametric import MahalanobisClassifier
from supervised_nonparametric import KnnClassifier
from data_manipulation import get_features, get_labels, get_subset, get_subset_rowinds, append_labels, load_heart, k_fold_split


def append_labels(features, labels):
    labels = reshape(labels, (-1, 1))
    # have to convert to array otherwise get_subset breaks
    return array(concatenate((features, labels), axis=-1))

def get_acc(classifier, X, y):
    # X and y are from test set
    te = append_labels(X, y)
    t0 = time.time()
    pred = classifier.predict(X)
    runtime = time.time() - t0
    truth = get_labels(te)
    i0 = get_subset_rowinds(te, 0)
    i1 = get_subset_rowinds(te, 1)
    acc_total = mean(pred == truth)
    acc0 = mean(pred[i0] == truth[i0])
    acc1 = mean(pred[i1] == truth[i1])
    return array([acc_total, acc0, acc1, runtime*1000]) # in ms

def conf_mat(classifier, X, y):
    # assumes classifier has already been trained
    te = append_labels(X, y)
    pred = classifier.predict(X)
    #truth = get_labels(te)
    i0 = get_subset_rowinds(te, 0)
    i1 = get_subset_rowinds(te, 1)
    c00 = sum(pred[i0] == 0)
    c01 = sum(pred[i0] == 1)
    c10 = sum(pred[i1] == 0)
    c11 = sum(pred[i1] == 1)
    return array([c00, c01, c10, c11])

def k_fold_conf_mat(classifier, X, y, k=5):
    num_classes = len(sort(unique(y)))
    if num_classes != 2:
        raise NotImplementedError

    cm = zeros(num_classes*num_classes, dtype=int) # stores counts, not probs
    for X_tr, y_tr, X_te, y_te in zip(*k_fold_split(X, y, k)):
        classifier.fit(X_tr, y_tr)
        cm += conf_mat(classifier, X_te, y_te)

    return cm

def roc_curve(classifier_constructor, X_tr, y_tr,  X_te, y_te, 
              hyperparams=linspace(0.001, 0.999, 101), label='', newfig=True):
    # hyperparams is an array of parameters that control the false error rate in some way
    FPR = zeros(len(hyperparams))
    TPR = zeros(len(hyperparams))
    for n, param in enumerate(hyperparams):
        classifier = classifier_constructor(param)
        TP, FN, FP, TN = k_fold_conf_mat(classifier, X_te, y_te)
        FPR[n] = FP/(FP + TN)
        TPR[n] = TP/(TP + FN)

    # putting in increasing order
    inds = argsort(FPR)
    FPR = FPR[inds]
    TPR = TPR[inds]

    if newfig:
        figure(figsize=(4,4))
    plot(FPR, TPR, '.', label=label)
    xlabel('FPR')
    ylabel('TPR')
    if label != '':
        legend(loc='lower right')
    grid(True)
    axis('square')
    title('ROC')
    xlim(0, 1)
    ylim(0, 1)
    tight_layout()


def main():
    
    X_tr, y_tr, X_te, y_te = load_heart(test_pct=50)

    roc_curve(MahalanobisClassifier,
              *load_heart(test_pct=50),
              label='MahalanobisClassifier',
              )

    roc_curve(KnnClassifier,
              *load_heart(test_pct=50),
              hyperparams=arange(500),
              label='KnnClassifier',
              )

    show()




if __name__ == '__main__':
    main()






