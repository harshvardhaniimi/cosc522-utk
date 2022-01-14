from pylab import *
from data_manipulation import load_heart, get_features, get_labels, get_subset
import sklearn.linear_model


def main():
    X_tr, y_tr, X_te, y_te = load_heart(test_pct=50)

    omp = sklearn.linear_model.OrthogonalMatchingPursuit()
    omp.fit(X_tr, y_tr)
    pred = (omp.predict(X_te) > 0.5).astype(int)
    print('OMP accuracy:', mean(pred == y_te))

    ridge = sklearn.linear_model.RidgeClassifier()
    ridge.fit(X_tr, y_tr)
    pred = ridge.predict(X_te)
    print('Ridge accuracy:', mean(pred == y_te))

    logistic = sklearn.linear_model.LogisticRegression()
    logistic.fit(X_tr, y_tr)
    pred = logistic.predict(X_te)
    print('Logistic accuracy:', mean(pred == y_te))

    passive_aggro = sklearn.linear_model.PassiveAggressiveClassifier()
    passive_aggro.fit(X_tr, y_tr)
    pred = passive_aggro.predict(X_te)
    print('Passive aggrressive accuracy:', mean(pred == y_te))


if __name__ == '__main__':
    main()
