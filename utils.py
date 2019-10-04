import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Activation Functions:
class Sigmoid(object):
    """ Sigmoid activation."""

    @staticmethod
    def activate(z):
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def derive(z):
        return Sigmoid.activate(z)*(1 - Sigmoid.activate(z))

class Tanh(object):
    """ Tanh activation. """

    @staticmethod
    def activate(z):
        return np.tanh(z)

    @staticmethod
    def derive(z):
        return 1 - Tanh.activate(z)**2

class ReLU(object):
    """ ReLU activation. """

    @staticmethod
    def activate(z):
        return np.maximum(0, z)

    @staticmethod
    def derive(z):
        return z > 0


# Cost Functions:
class CrossEntropy(object):
    """ Cross entropy cost function. """

    @staticmethod
    def compute(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        m = y.shape[1]
        return 1/m * np.sum(np.nan_to_num(-y*np.log(a)))

    @staticmethod
    def derive(a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        m = y.shape[1]
        return 1/m * (a-y)


# Miscellanious funtions:
def sliding(arr, size):
    """ Return a list of sliding windows over an array. """
    n_arr = len(arr)
    return [arr[i:i+size] for i in range(0, n_arr, size)]

def one_hot(target, num_classes):
    """ Return the target in one-hot form.
        y has to be a 1D array.
    """
    return np.eye(num_classes)[target].T

def plot_decision_regions(X, y, model, test_idx = None, resolution = 0.02):
    """A convenient function for plotting decision regions"""

    # Setup markers generator and color map.
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface.
    x1_min, x1_max = X[0, :].min() - 1, X[0, :].max() + 1
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]))
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all training samples.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[0, (y == cl).flatten()], y = X[1, (y == cl).flatten()], alpha = 0.8, c =
                cmap(idx), marker = markers[idx], label = cl)

    # Highlight test samples.
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(x = X_test[0, :], y = X_test[1, :], c = "",
                alpha = 1.0, linewidth = 1, marker = "o", s = 55,
                edgecolor = "black", label = "test set")


