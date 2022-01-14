from pylab import *

from data_manipulation import get_stats, load_heart, append_labels, get_subset

def evd(S):
    lambdas, P = eig(S)
    idx = lambdas.argsort()[::-1]
    Lambda = diag(lambdas[idx])
    P = P[:,idx]
    return P, Lambda

class PCA():
    def __init__(self, X, y):
        # X is training features, y are training labels

        tr = append_labels(X, y)
        _, Sigma = get_stats(tr)

        self._P, Lambda = evd(Sigma)
        self._lambdas = diag(Lambda)


    def __call__(self, X, ndims=1, error_rate=None):
        # X can be training or testing here

        if error_rate is not None:
            if error_rate < 0 or error_rate > 1:
                raise Exception('error_rate must be between 0 and 1, inclusive')
            threshold = 1.0 - error_rate
            error_fraction = cumsum(self._lambdas)/sum(self._lambdas)
            ndims = sum(error_fraction < threshold) + 1
        else:
            if ndims > X.shape[1]:
                raise Exception('ndims must be less than or equal to number of features')
            elif ndims < 1:
                raise Exception('ndims must be at least 1')

        return array(X@self._P[:,:ndims])

class FLD():
    def __init__(self, X, y):
        # X is training features, y are training labels

        tr = append_labels(X, y)
        _, Sigma = get_stats(append_labels(X, y))

        mu0, Sigma0 = get_stats(get_subset(tr, 0))
        mu1, Sigma1 = get_stats(get_subset(tr, 1))

        SW = Sigma0 + Sigma1
        w_pre = inv(SW)@(mu0 - mu1)
        self._w = w_pre/norm(w_pre)


    def __call__(self, X):

        return array(X@self._w)



def main():
    X_tr, y_tr, X_te, y_te = load_heart(test_pct=50)

    pca = PCA(X_tr, y_tr)
    fld = FLD(X_tr, y_tr)

    print(pca(X_te, ndims=4).shape)
    print(pca(X_te, error_rate=.05).shape)
    print(fld(X_te).shape)



if __name__ == '__main__':
    main()
