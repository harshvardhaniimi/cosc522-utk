from pylab import *

from data_manipulation import append_labels, load_heart, get_labels, get_features


def dist(x, y):
    # computes the distance matrix between two sets of row vectors using optimized ops
    if x.shape[1:] != y.shape[1:]:
        raise Exception('feature vectors must be same size')

    X = x.copy()[:,None,:]
    Y = y.copy()[None,:,:]
    return sqrt(sum((X - Y)**2, axis=-1))


def get_m_nearest_labels(dist_mat, labels, m):
    sorted_indices = argsort(dist_mat, axis=1)
    k_nearest_inds = sorted_indices[:,:m]
    k_nearest_labels = labels[k_nearest_inds]
    return k_nearest_labels



class KnnClassifier():
    def __init__(self, k=None):
        self._k = k

    def fit(self, X, y):

        if self._k is None:
            self._k = int(ceil(sqrt(X.shape[0])))

        self._tr = append_labels(X, y)

    def _dist(self, X):
        ''' computes the distance matrix from a set of feature vectors to every
        point in the training set using optimized array operations '''

        # strip off class labels and compute distance
        labels = get_labels(self._tr)
        dist_mat = dist(X, get_features(self._tr))

        return dist_mat

    def predict(self, X):
        ''' returns classification decision '''

        # assume labels are just zero or one
        #labels = self._tr[:,-1].astype(int)
        labels = get_labels(self._tr)
        if (labels.max() > 1) or (labels.min() < 0):
            raise Exception('labels needs to be zero or one')

        # find k nearest elements for each input test point
        dist_mat = self._dist(X)
        k_nearest_labels = get_m_nearest_labels(dist_mat, labels, m=self._k)
        threshold = self._k/2
        label_sums = k_nearest_labels.sum(axis=1)
        perturbation = 0.2*randint(2, size=len(label_sums)) - 0.1 # ensures result is random for ties
        decisions = (label_sums + perturbation) > threshold
        return decisions.astype(int)

def main():
    X_tr, y_tr, X_te, y_te = load_heart(test_pct=50)

    knn = KnnClassifier(k=4)
    knn.fit(X_tr, y_tr)
    print(mean(knn.predict(X_te) == y_te))


if __name__ == '__main__':
    main()
