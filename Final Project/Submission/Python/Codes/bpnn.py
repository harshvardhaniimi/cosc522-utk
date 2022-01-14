from pylab import *
from network import Network
from data_manipulation import load_heart, append_labels, get_subset
from performance_evaluation import get_acc

def int2onehot(x, N):
    y = zeros((N, 1), dtype=int)
    y[x,0] = 1
    return y

def split_data(data, train_pct=70):
    # divides data into training and test
    num_samples = data.shape[0]
    num_train = int(round(num_samples*train_pct/100))
    num_test = num_samples - num_train

    ind = permutation(num_samples)
    new_data = data[ind]

    data_train = []
    for row in new_data[:num_train]:
        feature = row[:-1]
        label = int(row[-1])
        data_train.append((feature[:,None], int2onehot(label, 2)))

    data_test = []
    for row in new_data[num_train:]:
        feature = row[:-1]
        label = int(row[-1])
        data_test.append((feature[:,None], label))

    return data_train, data_test


def load_data_wrapper(test_pct=50, supplied_data=None):
    if supplied_data is None:
        X_tr, y_tr, X_te, y_te = load_heart(test_pct=test_pct)
    else:
        X_tr, y_tr, X_te, y_te = supplied_data

    tr = append_labels(X_tr, y_tr)

    if X_te != [] and y_te != []:
        te = append_labels(X_te, y_te)

    tr0_train, tr0_test = split_data(get_subset(tr, 0), train_pct=70)
    tr1_train, tr1_test = split_data(get_subset(tr, 1), train_pct=70)

    tr_train = tr0_train + tr1_train
    tr_test = tr0_test + tr1_test

    if X_te == [] or y_te == []:
        return tr_train, tr_test, [], []
    else:
        te_train, te_test = split_data(te, train_pct=0)
        return tr_train, tr_test, te_train, te_test

def train(net):
    training_data, validation_data, _, test_data = load_data_wrapper()

    return net.SGD(training_data, epochs=1001, mini_batch_size=10, eta=0.01, test_data=validation_data, final_test_data=test_data, memory=100)

class NeuralNetwork():

    def __init__(self, layers=[12, 256, 2], mini_batch_size=10, eta=0.08, 
                 max_epochs=1001, early_stop_window=50):
        self.layers = layers

        self._net = Network(layers)
        self._mini_batch_size = mini_batch_size
        self._eta = eta
        self._max_epochs = max_epochs
        self._early_stop_window = early_stop_window

    def fit(self, X, y):

        training_data, validation_data, _, _ = load_data_wrapper(test_pct=50, 
                                                                 supplied_data=(X, y,
                                                                                [], [],
                                                                                ),
                                                                 )

        self._net.SGD(training_data, epochs=self._max_epochs, 
                      mini_batch_size=self._mini_batch_size, 
                      eta=self._eta, test_data=validation_data, 
                      memory=self._early_stop_window)

    def predict(self, X):
        return array([np.argmax(self._net.feedforward(x[:,None])) for x in X])



def main():
    X_tr, y_tr, X_te, y_te = load_heart(test_pct=50)

    nn = NeuralNetwork()
    nn.fit(X_tr, y_tr)
    print(nn.predict(X_te))
    print(get_acc(nn, X_te, y_te))


if __name__ == '__main__':
    main()
