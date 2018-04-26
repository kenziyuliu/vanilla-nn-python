import numpy as np

"""
TODO layers:
    BatchNorm
    WeightNorm

NOTE:
    Only use dropout in training
"""

class FullyConnected:
    def __init__(self, input_dim, output_dim, weight_initializer, use_bias=False):
        self.W_shape = (output_dim, input_dim)
        self.W = weight_initializer(self.W_shape)
        self.dW = None
        self.b = np.zeros(output_dim) if use_bias else None
        self.db = None
        self.input = None

    def forward(self, input):
        ''' Compute forward pass and save input '''
        self.input = input
        return np.matmul(self.weights, input)

    def backward(self, backproped_grad):
        '''
        Use back-propagated gradient to compute this layer's gradient
        This function saves dW and returns d(Loss)/d(input)
        '''
        self.dW = np.matmul(backproped_grad, self.input.T)
        dinput = np.matmul(self.W.T, backproped_grad)
        return dinput

    # NOTE: needs change
    def update(self, change):
        self.W += change


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)

    def backward(self, backproped_grad):
        deriv = np.where(self.input < 0, 0, 1)
        return backproped_grad * deriv


class LeakyReLU:
    def __init__(self, alpha=0.1):
        self.input = None
        if alpha <= 0 or alpha > 1:
            raise ValueError('LeakyReLU: alpha must be between 0 and 1')
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        return np.maximum(input, self.alpha * input)

    def backward(self, backproped_grad):
        deriv = np.where(self.input < 0, self.alpha, 1)
        return backproped_grad * deriv


class Softmax:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        '''
        Numerically more stable implementation of
        softmax function. Ref: http://cs231n.github.io/linear-classify/
        '''
        input -= np.max(input)  # maximum value is now 0, so exponentials wont break
        probabilities = np.exp(input) / np.sum(np.exp(input))
        self.input = input
        self.output = probabilities
        return probabilities

    def backward(self, backproped_grad):
        deriv = self.output - np.square(self.output)
        return backproped_grad * deriv


class CrossEntropyLoss:
    def __init__(self):
        self.y_pred = None   # Usually output from softmax layer
        self.y_true = None  # A real valued loss

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.log(y_pred[np.argmax(y_true)])   # -negative log probability of the right class

    def backward(self):
        return -self.y_true / self.y_pred


class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        self.input = None

    def forward(self, input):
        self.mask = np.random.binomial(1, self.rate, input.shape) / self.rate   # / rate so no change for prediction
        self.input = input
        return input * self.mask

    def backward(self, backproped_grad):
        deriv = backproped_grad * self.mask / self.rate # / rate so no change for prediction


def get_accuracy(y_pred, y_true):
    return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))


if __name__ == '__main__':
    """
    Test cases
    """
    print('relu and leaky relu:')
    values = np.random.rand(4,5) - .5
    print('value:\n', values)
    print('relu:', ReLU().forward(values))
    print('leakyrelu:', LeakyReLU(.1).forward(values))

    print('Softmax:')
    x = np.array([1,1,1,2], dtype='float64')
    print(x)
    print(Softmax().forward(x))

    print('Cross entropy loss:')
    y_pred = np.array([.1, .1, .1, .1, .1])
    y_true = np.array([0,0,1,0,0])
    print('y_pred: {}, y_true: {}'.format(y_pred, y_true))
    print('loss: {}'.format(CrossEntropyLoss().forward(y_pred, y_true)))

    print('Dropout:')
    x = np.arange(30).reshape(5, 6)
    print(x)
    print(Dropout(0.5).forward(x))
