import numpy as np
from initializers import *
from utils import *
import config

"""
TODO layers:
    BatchNorm
    WeightNorm

NOTE:
    Only use dropout in training
"""

class FullyConnected:
    def __init__(self, input_dim, output_dim, weight_initializer, use_bias=False, use_weight_norm=False):
        if use_weight_norm:
            # weight vector: length output_dim, there are input_dim number of it
            self.v_shape = (input_dim, output_dim)
            self.g_shape = (output_dim,)

            self.v = weight_initializer(self.v_shape)
            self.g = np.linalg.norm(self.v, axis=0)

            self.dv = None
            self.dg = None

        else:
            self.W_shape = (input_dim, output_dim)
            self.W = weight_initializer(self.W_shape)
            self.dW = None

        self.input = None
        self.b = np.zeros(output_dim)
        self.db = None
        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm

    def get_weight(self):
        if self.use_weight_norm:
            v_norm = np.linalg.norm(self.v, axis=0)
            return self.g * self.v / np.maximum(v_norm, config.EPSILON)
        else:
            return self.W

    def forward(self, input):
        '''
        Compute forward pass and save input
        input.shape = (batch x self.input_dim), output.shape = (batch x self.output_dim)
        '''
        self.input = input
        w = self.get_weight()
        return np.matmul(input, w) + (self.b if self.use_bias else 0)

    def backward(self, backproped_grad):
        '''
        Use back-propagated gradient to compute this layer's gradient
        This function saves dW and returns d(Loss)/d(input)
        backproped_grad.shape = (batch x self.output_dim), output.shape = (batch x self.input_dim)
        '''
        dweights = np.matmul(self.input.T, backproped_grad)                 # shape = (input_dim, output_dim)
        if self.use_weight_norm:
            v_norm = np.maximum(np.linalg.norm(v, axis=0), config.EPSILON)  # Clip for numerical stability
            self.dg = np.sum(dweights * self.v / v_norm, axis=0)            # Sum gradient since g was broadcasted
            self.dv = (self.g / v_norm * dweights) - (self.g * self.dg / np.square(v_norm) * self.v)
        else:
            self.dW = dweights

        if self.use_bias:
            # Sum gradient since bias was broadcasted
            self.db = np.sum(backproped_grad, axis=0)

        dinput = np.matmul(backproped_grad, self.get_weight().T)            # shape = (batch, input_dim)
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


if __name__ == '__main__':
    """
    Test cases
    """
    # print('relu and leaky relu:')
    input_val = np.random.randn(1,2)
    print('input_val:\n',input_val)

    grad_val = np.random.randn(1,3)
    print('grad_val:\n',grad_val)

    FC = FullyConnected(2,3,xavier_normal_init,use_weight_norm=True)
    FC_weight = FC.get_weight()
    # FC_weight = FC.W
    print('FC_weight:\n', FC_weight)
    print('FC_forward:\n', FC.forward(input_val))
    print('FC backward:\n',FC.backward(grad_val))
    print('FC.dv\n',FC.dv)
    print('FC.dg\n',FC.dg)

    # print('FC_length\n',np.linalg.norm(FC_weight.W, axis=0))
    print('relu:\n', ReLU().forward(input_val))
    print('leakyrelu:\n', LeakyReLU(.1).forward(input_val))
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
