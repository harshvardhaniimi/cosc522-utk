from pylab import *
from data_manipulation import load_heart, k_fold_split
from supervised_parametric import MahalanobisClassifier, QuadraticClassifier
from performance_evaluation import get_acc


def cross_val_acc(classifier, X, y, k=5):
    # performs k-fold cross-validation
    
    accs = zeros(4)
    for X_tr, y_tr, X_te, y_te in zip(*k_fold_split(X, y, k)):
        classifier.fit(X_tr, y_tr)
        accs += get_acc(classifier, X_te, y_te)/k # averaging over k folds

    return accs # (total_acc, acc_0, acc_1, runtime_ms) is format of output


def main():
    X, y, _, _ = load_heart(test_pct=0) # load all data into "training"
    P0 = 1 - mean(y)

    mah = MahalanobisClassifier(P0=P0)

    print(cross_val_acc(mah, X, y, k=5))


if __name__ == '__main__':
    main()
