import numpy as np
import random

np.random.seed(1)

def shuffle(x, y):
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    return x, y

class SLPerceptron:
    """
    Create a perceptron with a weight vector the same length
    as the feature vectors and a single bias. The weights and 
    biases are initialized with random gaussian deviates. 
    
    Constructor Params:
        input_len: number of features per input vector
        max_iter: training iterations
    """
    def __init__(self, input_len, max_iter = 100):
        assert type(input_len) is int and input_len > 0
        assert type(max_iter) is int and max_iter > 0
        self.max_iter = max_iter
        self.weights = np.random.normal(0, 1, input_len)
        self.biases = np.random.normal(0, 1, 1)
    def train(self, X, y):
        """
        The algorithm from the textbook
        """
        for i in range(self.max_iter):
            X, y = shuffle(X, y)
            for j, vector in enumerate(X):
                res = vector.dot(self.weights) + self.biases 
                prod = res * y[j]
                if prod <= 0:
                    self.weights += y[j] * vector
                    self.biases += y[j]
    def test(self, X, y):
        """
        Return the accuracy of the model against y
        """
        assert type(X) is np.ndarray
        assert type(y) is np.ndarray
        return np.mean(np.sign(X.dot(self.weights) + self.biases) == y)

    def __call__(self, x):
        """
        This class can be called as a function of the input 
        vector x
        """
        assert type(x) is np.ndarray
        assert len(x.shape) > 1
        assert x.shape[1] == self.weights.shape[0]
        res = x.dot(self.weights) + self.biases
        return np.sign(res)
    def __str__(self):
        n = len(self.weights)
        perc_string = "{}*x_{} + " * n
        w = np.round(self.weights, 2)
        b = self.biases[0]
        args = list(np.array([[w[i], i] for i in range(n)]).flatten())
        for i in range(1, len(args), 2):
            args[i] = int(args[i])
        return perc_string.format(*args) + str(np.round(b, 2)) + " = 0"