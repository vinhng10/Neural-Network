import numpy as np
from scipy.stats import bernoulli
from utils import *

class NeuralNet(object):
    """ First implementation of neural network !!! Fighting !!! """

    def __init__(self, sizes, activation='relu', cost=CrossEntropy,
            optimizer='adam', dropout=(0.9, 0.5), batch_norm=False,
            init='kaiming', eta=0.01, epochs=100, batch_size=64,
            beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.cost = cost
        self.optimizer = optimizer
        self.p_in = dropout[0]
        self.p_hid = dropout[1]
        self.batch_norm = batch_norm
        self.init = init
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        if activation == 'sigmoid':
            self.activation = Sigmoid
        elif activation == 'tanh':
            self.activation = Tanh
        else:
            self.activation = ReLU

    def initialize(self):
        """ For sigmoid & tanh activation ----> Xavier.
            For ReLU & Leaky ReLU activation ----> Kaiming.
            Lth layer has shape [l, l-1].
        """
        # Momentum initialization:
        self.m_dws = [np.zeros((y, x))
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.m_dbs = [np.zeros((y, 1)) for y in self.sizes[1:]]

        # RMSProp initialization:
        self.s_dws = [np.zeros((y, x))
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.s_dbs = [np.zeros((y, 1)) for y in self.sizes[1:]]

        # Kaiming initialization:
        self.weights = [np.random.randn(y, x)*np.sqrt(2/x)
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def get_masks(self, phase):
        """ Sample masks for the net's layers.

            At training time, the network is thinned by halving
            the number of hidden units, sampling input with
            probability close to 1, and reserving all output units.

            At test time, the full network is used, but its
            weights and biases are multiplied with the sampling
            probability to ensure that their expected values are the
            same as that of any thinned networks' weights and biases.

            If 'dropout' is set to 'False', generated masks are matrices
            filled only with 1.0.
        """
        if phase == 'train':
            in_mask = [bernoulli.rvs(self.p_in, size=(self.sizes[0], 1))]
            hid_mask = [bernoulli.rvs(self.p_hid, size=(l, 1))
                    for l in self.sizes[1: -1]]
            return in_mask + hid_mask
        else:
            in_mask = [np.full((self.sizes[0], 1), self.p_in)]
            hid_mask = [np.full((l, 1), self.p_hid) for l in self.sizes[1:-1]]
            return in_mask + hid_mask

    def forward_prop(self, a, masks):
        """ Forward propagation to compute the cost.
            At each layer, store the linear combination
            and the activation for later computation of
            back propagation.
        """
        z_cache = []
        a_cache = []
        for w, b, m in zip(self.weights, self.biases, masks):
            a = a * m
            a_cache.append(a)
            z = w @ a + b
            z_cache.append(z)
            a = self.activation.activate(z)
        # Softmax output layer:
        a_cache.append(np.exp(z) / np.sum(np.exp(z), axis=0))
        return z_cache, a_cache

    def total_cost(self, a, y, masks):
        """ Compute the cost with the given input a. """
        z_cache, a_cache = self.forward_prop(a, masks)
        return self.cost.compute(a_cache[-1], y)

    def backward_prop(self, a, y, masks):
        """ Propagate backward through the network to compute
            the gradient of each weight and bias.
        """
        dw_cache = [0 for w in self.weights]
        db_cache = [0 for b in self.biases]
        z_cache, a_cache = self.forward_prop(a, masks)
        z_cache = [0] + z_cache             # Just for convenience
        dz = self.cost.derive(a_cache[-1], y)
        for l in range(1, self.num_layers):
            dw = dz @ a_cache[-l-1].T
            db = np.sum(dz, axis=1, keepdims=True)
            dw_cache[-l] = dw
            db_cache[-l] = db
            da = self.weights[-l].T @ dz
            dz = da * self.activation.derive(z_cache[-l-1]) * masks[-l]
        return dw_cache, db_cache

    def update_params(self, a, y, masks, eta, beta1, beta2, epsilon):
        """ Update each weight and bias to reduce the cost. """
        dw_cache, db_cache = self.backward_prop(a, y, masks)

        # Momentum:
        self.m_dws = [(beta1*m_dw + (1-beta1)*dw)
                for m_dw, dw in zip(self.m_dws, dw_cache)]
        self.m_dbs = [(beta1*m_db + (1-beta1)*db)
                for m_db, db in zip(self.m_dbs, db_cache)]

        # RMSProp:
        self.s_dws = [(beta2*s_dw + (1-beta2)*(dw**2))
                for s_dw, dw in zip(self.s_dws, dw_cache)]
        self.s_dbs = [(beta2*s_db + (1-beta2)*(db**2))
                for s_db, db in zip(self.s_dbs, db_cache)]

        # Momentum + RMSProp:
        w_adams = [m_dw / (np.sqrt(s_dw) + epsilon)
                for m_dw, s_dw in zip(self.m_dws, self.s_dws)]
        b_adams = [m_db / (np.sqrt(s_db) + epsilon)
                for m_db, s_db in zip(self.m_dbs, self.s_dbs)]

        # Update parameters:
        self.weights = [w - eta*w_adam
                for w, w_adam in zip(self.weights, w_adams)]
        self.biases = [b - eta*b_adam
                for b, b_adam in zip(self.biases, b_adams)]

    def fit(self, X, y):
        """ Fit the network to the traning data using SGD. """
        self.initialize()
        idx = np.arange(y.shape[1])
        self.costs = []
        for i in range(self.epochs):
            np.random.shuffle(idx)
            batch_indices = sliding(idx, self.batch_size)
            for b_idx in batch_indices:
                batch_X, batch_y = X[:, b_idx], y[:, b_idx]
                masks = self.get_masks('train')
                self.update_params(batch_X, batch_y, masks,
                        self.eta, self.beta1, self.beta2, self.epsilon)
                if i % 100 == 0:
                    self.costs.append(self.total_cost(batch_X, batch_y, masks))

    def predict(self, X):
        """ Predict the class of the input. """
        masks = self.get_masks('test')
        _, a_cache = self.forward_prop(X, masks)
        return np.argmax(a_cache[-1], axis=0)

    def accuracy(self, X, y):
        """ Return the accuracy of the net on the given dataset. """
        check = self.predict(X) == y.flatten()
        return 100.0 * sum(check) / len(check)


class ConvoluteNet(object):
    """ First implementation of convolutional network !!! Fighting !!! """


class RecurrentNet(object):
    """ First implementation of recurrent network !!! Fighting !!! """



