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
        self.W_shape = (input_dim, output_dim)
        self.W = weight_initializer(self.W_shape)
        self.dW = None
        self.b = np.zeros(output_dim) if use_bias else None
        self.db = None
        self.input = None

    def forward(self, input):
        '''
        Compute forward pass and save input
        input.shape = (batch x self.input_dim), output.shape = (batch x self.output_dim)
        '''
        self.input = input
        return np.matmul(input, self.weights) + (0 if self.b is None else self.b)

    def backward(self, backproped_grad):
        '''
        Use back-propagated gradient to compute this layer's gradient
        This function saves dW and returns d(Loss)/d(input)
        backproped_grad.shape = (batch x self.output_dim), output.shape = (batch x self.input_dim)
        '''
        self.dW = np.matmul(self.input.T, backproped_grad)
        # As bias was broadcast during forward, now take the average of backproped grad along *batch size*
        self.db = np.average(backproped_grad, axis=0) if self.b is not None else 0
        dinput = np.matmul(backproped_grad, self.W.T)
        return dinput

    # NOTE: needs change
    def update(self, change):
        self.W += change


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input):
        ''' input.shape = output.shape = (batch x input_dims) '''
        self.input = input
        return np.maximum(input, 0)

    def backward(self, backproped_grad):
        deriv = np.where(self.input < 0, 0, 1)
        return backproped_grad * deriv


class LeakyReLU:
    def __init__(self, alpha=0.1):
        if alpha <= 0 or alpha > 1:
            raise ValueError('LeakyReLU: alpha must be between 0 and 1')
        self.alpha = alpha
        self.input = None

    def forward(self, input):
        ''' input.shape = output.shape = (batch x input_dims) '''
        self.input = input
        return np.maximum(input, self.alpha * input)

    def backward(self, backproped_grad):
        deriv = np.where(self.input < 0, self.alpha, 1)
        return backproped_grad * deriv


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, input):
        '''
        Numerically more stable implementation of
        softmax function. Ref: http://cs231n.github.io/linear-classify/
        input.shape = output.shape = (batch x input_dims)
        '''
        # maximum value is now 0 along each training example, so exponentials wont break
        input -= np.max(input, axis=1, keepdims=True)
        exponentials = np.exp(input)
        self.output = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        return self.output

    def backward(self, backproped_grad):
        deriv = self.output - np.square(self.output)
        return backproped_grad * deriv


class CrossEntropyLoss:
    def __init__(self):
        self.y_pred = None   # Usually output from softmax layer
        self.y_true = None  # A real valued loss for each training example

    def forward(self, y_pred, y_true):
        ''' y_pred.shape = y_true.shape = (batch x num_classes), output.shape = (batch,) '''
        self.y_pred = y_pred
        self.y_true = y_true
        # -negative log probability of the right class for each example in the batch
        right_class_probs = y_pred[np.arange(len(y_pred)), np.argmax(y_true, axis=1)]
        return -np.log(right_class_probs)

    def backward(self):
        return -self.y_true / self.y_pred


class Dropout:
    def __init__(self, rate):
        if rate < 0 or rate >= 1:
            raise ValueError('Dropout: dropout rate must be >= 0 and < 1')
        self.rate = rate
        self.mask = None
        self.input = None

    def forward(self, input):
        ''' input.shape = output.shape = (batch x input_dims) '''
        self.mask = np.random.binomial(1, self.rate, input.shape) / self.rate   # divide rate so no change for prediction
        self.input = input
        return input * self.mask

    def backward(self, backproped_grad):
        ''' backproped_grad.shape = (batch x input_dims) '''
        deriv = backproped_grad * self.mask / self.rate # divide rate so no change for prediction

        
def get_accuracy(y_pred, y_true):
    return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))


if __name__ == '__main__':
    """
    Test cases
    """
    print('relu and leaky relu:')
    values = np.random.rand(4,5) - .5
    print('value:\n', values)
    print('relu:\n', ReLU().forward(values))
    print('leakyrelu:\n', LeakyReLU(.1).forward(values))
    print()
    print('Softmax:')
    x = np.expand_dims(np.array([1,1,1,2], dtype='float64'), axis=0)  # Mock the batch dimension
    print(x)
    print(Softmax().forward(x))
    print()
    print('Cross entropy loss:')
    y_pred = np.expand_dims(np.array([.1, .1, .1, .1, .1]), axis=0)
    y_true = np.expand_dims(np.array([0,0,1,0,0]), axis=0)
    print('y_pred:\n {} \ny_true:\n {}'.format(y_pred, y_true))
    print('loss:\n {}'.format(CrossEntropyLoss().forward(y_pred, y_true)))
    print()
    print('Dropout:')
    x = np.arange(30).reshape(5, 6)
    print(x)
    print(Dropout(0.5).forward(x))


