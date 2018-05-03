import config
import numpy as np

# Little factory method
def get_optimizer(name):
    name = name.lower()
    if name == 'sgd':
        return SGD(config.LEARNING_RATE)
    elif name == 'adam':
        return Adam(config.LEARNING_RATE)
    else:
        raise ValueError('Unsupported Optimizer: "{}"'.format(name))


class SGD:
    def __init__(self, learning_rate, momentum=0.9, nesterov=True):
        self.learning_rate = learning_rate
        self.v = []
        self.momentum = momentum
        self.use_nesterov = nesterov

    def init_shape(self, shape_list):
        self.v = [np.zeros(shape, dtype='float64') for shape in shape_list]

    def optimize(self, params_gradient):
        ''' Expects tupled parameters and their gradients e.g. [(w, dw), (b, db), ...] '''
        assert len(params_gradient) == len(self.v)
        for i in range(len(params_gradient)):
            parameter, gradient = params_gradient[i]
            assert self.v[i].shape == parameter.shape == gradient.shape

            if self.use_nesterov:
                change = self.v[i]
                self.v[i] = self.momentum * self.v[i] + self.learning_rate * gradient
                change = self.momentum * change - (1.0 + self.momentum) * self.v[i]
            else:
                change = self.momentum * self.v[i] - self.learning_rate * gradient
                self.v[i] = change

            parameter += change


class Adam:
    def __init__(self, learning_rate=0.001, B1=0.9, B2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.B1 = B1
        self.B2 = B2
        self.epsilon = epsilon
        self.mt = []
        self.vt = []
        self.t = 0

    def init_shape(self, shape_list):
        for shape in shape_list:
            self.mt.append(np.zeros(shape))
            self.vt.append(np.zeros(shape))

    def optimize(self, params_gradient):
        ''' Expects tupled parameters and their gradients e.g. [(w, dw), (b, db), ...] '''
        assert len(params_gradient) == len(self.mt) == len(self.vt)
        self.t += 1
        for i in range(len(params_gradient)):
            parameter, gradient = params_gradient[i]
            assert parameter.shape == gradient.shape == self.mt[i].shape == self.vt[i].shape
            # Adam update
            self.mt[i] = self.mt[i] * self.B1 + (1. - self.B1) * gradient
            self.vt[i] = self.vt[i] * self.B2 + (1. - self.B2) * (gradient * gradient)
            mt_hat = self.mt[i] / (1. - (self.B1 ** self.t))
            vt_hat = self.vt[i] / (1. - (self.B2 ** self.t))
            change = (-self.learning_rate) * mt_hat / (np.sqrt(vt_hat) + self.epsilon)

            parameter += change
            
