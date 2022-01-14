from pylab import *

from data_manipulation import get_subset, append_labels, load_heart, get_stats


class Classifier():
    def __init__(self, P0=0.5):
        self._P0 = P0

    def _g0(self, x):
        return self._b0.T@x + self._c0

    def _g1(self, x):
        return self._b1.T@x + self._c1

    def predict(self, X):
        transposed_X = X.T
        return array((self._g0(transposed_X) - self._g1(transposed_X) < 0).astype(int)).flatten()


class MinDistClassifier(Classifier):

    def fit(self, X, y):

        P = array([self._P0, 1 - self._P0])
        tr = append_labels(X, y)

        # saving training data for later
        self._tr = tr.copy()
        # compute class means
        mu0 = get_stats(get_subset(tr, 0))[0]
        mu1 = get_stats(get_subset(tr, 1))[0]
        # compute overall variance

        #sigmasq = get_total_variance(tr)
        Sig = get_stats(tr)[1]
        sigmasq = mean(diag(Sig))

        # precompute left-multiplication matrix (A) + bias (b) for cases 0, 1 for speed
        self._b0 = mu0/sigmasq
        self._b1 = mu1/sigmasq
        self._c0 = -mu0.T@mu0/(2*sigmasq) + log(P[0])
        self._c1 = -mu1.T@mu1/(2*sigmasq) + log(P[1])


class MahalanobisClassifier(Classifier):

    def fit(self, X, y):
        
        cov_method = 'avg'
        P = array([self._P0, 1.0 - self._P0])
        tr = append_labels(X, y)

        if abs(sum(P) - 1) > 1e-3:
            raise Exception('prior probabilities must sum to one')

        # saving training data for later
        self._tr = tr.copy()
        # compute class means
        mu0 = get_stats(get_subset(tr, 0))[0]
        mu1 = get_stats(get_subset(tr, 1))[0]
        # compute overall covariance matrix (inverse)
        if cov_method == 'avg':
            Sigma0 = get_stats(get_subset(tr, 0))[1]
            Sigma1 = get_stats(get_subset(tr, 1))[1]
            Sigma = (Sigma0 + Sigma1)/2.
        elif cov_method == 'overall':
            Sigma = get_stats(tr)[1]
        elif cov_method == 'Sigma0':
            Sigma = get_stats(get_subset(tr, 0))[1]
        elif cov_method == 'Sigma1':
            Sigma = get_stats(get_subset(tr, 1))[1]
        else:
            raise Exception('invalid cov_method choice')

        Sigma_inv = inv(Sigma)

        # precompute left-multiplication matrix (A) + bias (b) for cases 0, 1 for speed
        self._b0 = Sigma_inv@mu0
        self._b1 = Sigma_inv@mu1
        self._c0 = -0.5*mu0.T@Sigma_inv@mu0 + log(P[0])
        self._c1 = -0.5*mu1.T@Sigma_inv@mu1 + log(P[1])

class QuadraticClassifier(Classifier):
    def fit(self, X, y):

        P = array([self._P0, 1.0 - self._P0])
        tr = append_labels(X, y)

        if abs(sum(P) - 1) > 1e-3:
            raise Exception('prior probabilities must sum to one')

        # saving training data for later
        self._tr = tr.copy()
        # compute class means
        mu0 = get_stats(get_subset(tr, 0))[0]
        mu1 = get_stats(get_subset(tr, 1))[0]
        # compute class covariance matrices (inverse)
        Sigma0 = get_stats(get_subset(tr, 0))[1]
        Sigma1 = get_stats(get_subset(tr, 1))[1]
        Sigma0_inv = inv(Sigma0)
        Sigma1_inv = inv(Sigma1)

        # precompute quadratic kernel (C), left matrix (A), and 
        # bias (b) for cases 0, 1 for speed
        self._A0 = -Sigma0_inv/2
        self._A1 = -Sigma1_inv/2

        self._b0 = Sigma0_inv@mu0
        self._b1 = Sigma1_inv@mu1

        self._c0 = -0.5*mu0.T@Sigma0_inv@mu0 - 0.5*log(det(Sigma0)) + log(P[0])
        self._c1 = -0.5*mu1.T@Sigma1_inv@mu1 - 0.5*log(det(Sigma1)) + log(P[1])

    # have to redefine these to add quadratic term
    def _g0(self, x):
        # efficient but ugly way
        left = array(x)
        right = array(self._A0@x)
        return (left*right).sum(axis=0) + self._b0.T@x + self._c0

    def _g1(self, x):
        # efficient but ugly way
        left = array(x)
        right = array(self._A1@x)
        return (left*right).sum(axis=0) + self._b1.T@x + self._c1


def main():
    X_tr, y_tr, X_te, y_te = load_heart(test_pct=50)
    P0 = 1 - mean(y_tr)

    mindist = MinDistClassifier(P0=P0)
    mindist.fit(X_tr, y_tr)
    print(mean(mindist.predict(X_te) == y_te))

    mah = MahalanobisClassifier(P0=P0)
    mah.fit(X_tr, y_tr)
    print(mean(mah.predict(X_te) == y_te))

    quad = QuadraticClassifier(P0=P0)
    quad.fit(X_tr, y_tr)
    print(mean(quad.predict(X_te) == y_te))


if __name__ == '__main__':
    main()
