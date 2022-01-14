from pylab import *

def k_fold_split(X, y, k=5, rng_seed=0):
    num_samples = len(X)

    seed(rng_seed)
    inds = permutation(arange(num_samples))
    X_shuffled = X[inds]
    y_shuffled = y[inds]
    
    num_test = int(round(num_samples/k))

    X_tr = []
    y_tr = []
    X_te = []
    y_te = []

    for n in range(k):
        start = n*num_test
        if n == k - 1:
            end = num_samples
        else:
            end = (n + 1)*num_test

        X_tr += [concatenate((X_shuffled[:start], X_shuffled[end:]), axis=0)]
        y_tr += [concatenate((y_shuffled[:start], y_shuffled[end:]), axis=0)]

        X_te += [X_shuffled[start:end]]
        y_te += [y_shuffled[start:end]]

    return X_tr, y_tr, X_te, y_te

def get_stats(data):
    # estimates the first order (mean) and second order (covariance matrix) statistics
    X = data[:,:-1] # just the feature partition of matrix
    N = X.shape[0]
    mu = matrix(X.mean(axis=0)).T # make into col vector
    mu_repeated = tile(mu.T, (N, 1))
    Y = X - mu_repeated
    Sigma = (Y.T@Y)/(N - 1) # computing sample covariance matrix; @ is new syntax for matrix mult.
    return mu, Sigma

def get_limits(data):
    # assumes 2d
    xmin = data[:,0].min()
    xmax = data[:,0].max()
    ymin = data[:,1].min()
    ymax = data[:,1].max()
    return xmin, xmax, ymin, ymax

def plot_image(x, y, z, figsize=None, label=None,
               interpolation='none',
               cticks=None,
               cmap=cm.gray,
               ):

    if ndim(x) == 1 and ndim(y) == 1:
        if len(x) != z.shape[0] or len(y) != z.shape[1]:
            raise Exception("dimensions don't match")
    else:
        raise NotImplementedError

    dx = median(diff(sort(unique(x))))
    dy = median(diff(sort(unique(y))))

    if type(figsize) != type(None):
        fig = figure(figsize=figsize)
    else:
        fig = figure()

    imshow(
            z.T,
            aspect='auto',
            origin='lower',
            interpolation=interpolation,
            extent=(
                    x.min() - dx/2.,
                    x.max() + dx/2.,
                    y.min() - dy/2.,
                    y.max() + dy/2.,
                    ),
            cmap=cmap,
            )
    if cticks is None:
        cb = colorbar()
    else:
        cb = colorbar(ticks=cticks)

    if label != None:
        cb.set_label(label)
    return cb

def get_subset(data, class_label):
    # returns the rows of the data matrix 
    return data[get_subset_rowinds(data, class_label), :]

def get_subset_rowinds(data, class_label):
    # returns the indicies of the rows of the data matrix 
    return matrix(data[:,-1] == class_label).A1

def append_labels(X, y):
    return concatenate((X, y[:,None]), axis=1)

def get_labels(data):
    # returns just the label column
    return data[:,-1].copy().astype(int)

def get_features(data):
    # returns just the feature vectors
    return data[:,:-1].copy()

def converter(in_str):
    # function used to convert the 'Yes'/'No' labels to integers
    if in_str == b'Yes':
        return 1
    elif in_str == b'No':
        return 0
    else:
        raise Exception('invalid string')

def load_synth():
    synth_tr = loadtxt('synth.tr', skiprows=1)
    synth_te = loadtxt('synth.te', skiprows=1)

    return synth_tr, synth_te

def load_pima():
    # using "converter" function to map Yes/No to 0/1
    pima_tr = loadtxt('pima.tr', skiprows=1, converters={7: converter})
    pima_te = loadtxt('pima.te', skiprows=1, converters={7: converter})

    # normalizing pima data
    n_feat = pima_tr.shape[1] - 1
    n_samples = pima_tr.shape[0]
    for feature_num in range(n_feat):
        # estimating statistics
        mu = mean(pima_tr[:, feature_num])
        sigma = std(pima_tr[:, feature_num], ddof=1) # sample std, not pop
        # normalizing
        pima_tr[:, feature_num] -= mu
        pima_te[:, feature_num] -= mu
        pima_tr[:, feature_num] /= sigma
        pima_te[:, feature_num] /= sigma

    return pima_tr, pima_te

def load_heart(test_pct=50):
    data = loadtxt('heart_failure_clinical_records_dataset.csv', delimiter = ',', skiprows=1)

    # normalizing heart data
    n_feat = data.shape[1] - 1
    n_samples = data.shape[0]
    for feature_num in range(n_feat):
        # estimating statistics
        mu = mean(data[:, feature_num])
        sigma = std(data[:, feature_num], ddof=1) # sample std, not pop
        # normalizing
        data[:, feature_num] -= mu
        data[:, feature_num] /= sigma

    # randomly shuffling data
    #seed(0) # for predictable results; comment out if want random
    inds = permutation(n_samples)
    data_shuffled = data[inds]
    num_train = int(round((100 - test_pct)/100*n_samples))

    tr = data_shuffled[:num_train]
    te = data_shuffled[num_train:]

    return get_features(tr), get_labels(tr), get_features(te), get_labels(te)


def main():
    print(load_heart())


if __name__ == '__main__':
    main()
