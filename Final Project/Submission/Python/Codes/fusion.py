from pylab import *

from supervised_parametric import MahalanobisClassifier, QuadraticClassifier
from data_manipulation import get_features, get_labels, load_heart
from performance_evaluation import conf_mat, get_acc

from itertools import product


class Fuser():
    def __init__(self, classifiers):
        # classifiers is a list of classifier functions

        self._classifiers = classifiers


class BKS(Fuser):

    def fit(self, X, y):
        num_classes = len(unique(y))
        if num_classes != 2:
            raise NotImplementedError('currently only supports two class problems')

        # train all clasifiers
        decisions = []
        for classifier in self._classifiers:
            classifier.fit(X, y)
            decisions += [classifier.predict(X)]

        # have to build up BKS self._table sample by sample
        self._table = {}
        for n in range(len(y)):
            truth = y[n]
            chosen_labels = tuple([decisions[m][n] for m in range(len(self._classifiers))])

            if not (chosen_labels in self._table.keys()):
                self._table[chosen_labels] = zeros(2, dtype=int).tolist()

            self._table[chosen_labels][truth] += 1

    def predict(self, X):

        decisions = [classifier.predict(X) for classifier in self._classifiers]
        
        outputs = zeros(len(X), dtype=int)
        for n, sample in enumerate(X):
            chosen_labels = tuple([decisions[m][n] for m in range(len(self._classifiers))])
            try:
                outputs[n] = argmax(self._table[chosen_labels])
            except KeyError: # table is empty; pick random
                outputs[n] = randint(2)

        return outputs

class NaiveBayes(Fuser):
    def fit(self, X, y):

        num_classes = len(unique(y))
        num_classifiers = len(self._classifiers)
        if num_classes != 2:
            raise NotImplementedError('currently only supports two class problems')

        Nk = zeros(num_classes, dtype=int)
        Nk[0] = sum(y == 0)
        Nk[1] = sum(y == 1)
        N = sum(Nk)

        cms = []
        for classifier in self._classifiers:
            TP, FN, FP, TN = conf_mat(classifier, X, y)
            cm = array([[TP, FN],
                        [FP, TN]])
            cms.append(cm)

        omegas = range(num_classes)

        # filling up unnormalized posterior probability matrix
        P_omega_k_s = zeros((num_classes,)*(num_classifiers + 1))
        for k in omegas:
            for s in product(omegas, repeat=num_classifiers):
                val = 1
                for n in range(num_classifiers):
                    val *= cms[n][k, s[k]]
                P_omega_k_s[(k,) + s] = val/Nk[k]

        # normalizing posterior probabilities
        for s in product(omegas, repeat=num_classifiers):
            total_prob = 0.
            for k in omegas:
                total_prob += P_omega_k_s[(k,) + s]

            for k in omegas:
                if total_prob:
                    P_omega_k_s[(k,) + s] /= total_prob

        # building up decision table
        self._table = {}

        for s in product(omegas, repeat=num_classifiers):
            self._table[s] = argmax(P_omega_k_s, axis=0)[s]


    def predict(self, X):

        decisions = [classifier.predict(X) for classifier in self._classifiers]

        outputs = zeros(len(X), dtype=int)
        for n in range(len(X)):
            chosen_labels = tuple([decisions[m][n] for m in range(len(self._classifiers))])
            try:
                outputs[n] = self._table[chosen_labels]
            except KeyError: # table is empty; pick random
                outputs[n] = randint(2)

        return outputs


def test_acc(classifier, te):
    return mean(classifier(te) == get_labels(te))


def main():
    X_tr, y_tr, X_te, y_te = load_heart(test_pct=50)

    P0 = 1 - mean(y_tr)

    mah = MahalanobisClassifier(P0=P0)
    quad = QuadraticClassifier(P0=P0)

    bks = BKS((mah, quad))
    bks.fit(X_tr, y_tr)
    bks.predict(X_te)

    nb = NaiveBayes((mah, quad))
    nb.fit(X_tr, y_tr)
    nb.predict(X_te)

    print('mah:', get_acc(mah, X_te, y_te))
    print('quad:', get_acc(quad, X_te, y_te))
    print('bks:', get_acc(bks, X_te, y_te))
    print('nb:', get_acc(nb, X_te, y_te))


if __name__ == '__main__':
    main()
